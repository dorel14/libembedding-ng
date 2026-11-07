/*
 * libembedding - benchmarks/bench_text_onnx.cpp
 * Dense text throughput on the ONNX backend: library vs bare ONNX Runtime.
 *
 * Separates "cost of the library" from "limit of ORT / hardware" on one model:
 *   --mode lib   public text API (tokenise + pad + Run + pool + normalise)
 *   --mode raw   same model, same token ids, same batches (sorted by length,
 *                padded to the batch max), straight on the ORT C API: only
 *                Run() is timed, tokenisation is done once beforehand and
 *                reported separately.
 *   --mode stages  replays the library pipeline step by step and prints the
 *                time of each stage (tokenise, tensors, Run, pool+norm).
 *
 * Corpora (n texts, cycled lengths in words):
 *   mixed = 8/24/48/96/160   short = ~10   long = ~150   single = 1 text, latency
 *
 * Usage (Release, configure with -DLIBEMBEDDING_BUILD_BENCHMARKS=ON):
 *   bench_text_onnx.exe --mode lib --corpus mixed --threads 0 --batch 256
 * Options: --model-dir <dir containing model.onnx + tokenizer.json> (default:
 *          the Qdrant all-MiniLM-L6-v2 entry of the local cache),
 *          --workers W (one session per worker), --threads T (intra-op, 0 = ORT
 *          default), --batch B, --runs N (median + min/max), --n N texts,
 *          --nospin (disable ORT intra-op spinning, raw mode),
 *          --dump <file> (lib mode: write the embeddings as raw float32).
 * One JSON line on stdout. The model must already be cached (nothing downloaded).
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {

double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

const char* kWords[] = {
    "retrieval", "augmented", "generation", "combines", "search", "with", "language",
    "models", "dense", "vectors", "capture", "meaning", "while", "sparse", "weights",
    "keep", "exact", "terms", "rerankers", "score", "query", "document", "pairs",
    "latency", "matters", "for", "interactive", "systems", "throughput", "for", "batch",
    "indexing", "of", "large", "collections", "on", "commodity", "hardware",
};
const int kNumWords = (int)(sizeof(kWords) / sizeof(kWords[0]));

std::vector<std::string> make_texts(const std::string& corpus, int n) {
    std::vector<int> lengths;
    if (corpus == "short") lengths = {8, 10, 12};
    else if (corpus == "long") lengths = {140, 150, 160};
    else if (corpus == "single") { lengths = {24}; n = 1; }
    else lengths = {8, 24, 48, 96, 160};
    std::vector<std::string> out;
    for (int i = 0; i < n; i++) {
        int len = lengths[i % lengths.size()];
        std::string s;
        for (int j = 0; j < len; j++) {
            if (j) s += ' ';
            s += kWords[(i * 11 + j * 3) % kNumWords];
        }
        out.push_back(s);
    }
    return out;
}

std::string default_model_dir() {
    const char* home = std::getenv("USERPROFILE");
    if (!home) home = std::getenv("HOME");
    return std::string(home ? home : ".") + "/.cache/libembedding/models--Qdrant-all-MiniLM-L6-v2-onnx";
}

struct Args {
    std::string mode = "lib", corpus = "mixed", model_dir, dump;
    int workers = 1, threads = 0, batch = 256, runs = 7, n = 256;
    bool nospin = false;
};

void report(const Args& a, const char* mode, const std::vector<double>& ms, int n_texts,
            long total_tokens, long padded_tokens, const char* extra) {
    double med = median(ms);
    double mn = *std::min_element(ms.begin(), ms.end());
    double mx = *std::max_element(ms.begin(), ms.end());
    std::printf("{\"mode\":\"%s\",\"corpus\":\"%s\",\"n\":%d,\"workers\":%d,\"threads\":%d,"
                "\"batch\":%d,\"runs\":%d,\"tokens_total\":%ld,\"tokens_mean\":%.1f,"
                "\"tokens_padded\":%ld,\"median_ms\":%.2f,\"min_ms\":%.2f,\"max_ms\":%.2f,"
                "\"docs_s\":%.2f,\"docs_s_best\":%.2f,\"tok_s\":%.0f%s}\n",
                mode, a.corpus.c_str(), n_texts, a.workers, a.threads, a.batch, (int)ms.size(),
                total_tokens, (double)total_tokens / n_texts, padded_tokens, med, mn, mx,
                n_texts * 1000.0 / med, n_texts * 1000.0 / mn, total_tokens * 1000.0 / med,
                extra);
}

/* ------------------------------------------------------------------ lib */
int run_lib(const Args& a, const std::vector<std::string>& texts) {
    lembed_text_options_t o = lembed_text_options_default();
    o.model = (lembed_text_model_t)lembed_resolve_text_model("sentence-transformers/all-MiniLM-L6-v2");
    o.offline = 1;
    o.show_download_progress = 0;
    if (a.threads > 0) o.num_threads = a.threads;
    o.batch_size = a.batch;
    std::vector<lembed_text_embedding_t*> ctx(a.workers, nullptr);
    for (int i = 0; i < a.workers; i++)
        if (lembed_text_embedding_create(&o, &ctx[i]) != LEMBED_OK) {
            std::fprintf(stderr, "create failed: %s\n", lembed_last_error());
            return 1;
        }
    std::vector<const char*> ct;
    for (auto& s : texts) ct.push_back(s.c_str());
    const int n = (int)texts.size();

    /* token count (tokeniser of ctx 0, same as the library's) */
    lembed::detail::TokenizerWrapper tk;
    tk.load_from_file((a.model_dir.empty() ? default_model_dir() : a.model_dir) + "/tokenizer.json", 512);
    long total_tokens = 0;
    for (auto& s : texts) total_tokens += (long)tk.encode(s).size();

    std::vector<std::vector<float>> out(a.workers);
    auto run = [&]() -> bool {
        std::vector<int> ok(a.workers, 0);
        std::vector<std::thread> th;
        int per = n / a.workers;
        for (int w = 0; w < a.workers; w++) {
            int start = w * per;
            int cnt = (w == a.workers - 1) ? n - start : per;
            th.emplace_back([&, w, start, cnt]() {
                lembed_embeddings_t r = {0};
                ok[w] = lembed_text_embedding_embed(ctx[w], ct.data() + start, cnt, 0, &r) == LEMBED_OK;
                if (ok[w] && !a.dump.empty()) out[w].assign(r.data, r.data + (size_t)cnt * r.dim);
                lembed_embeddings_free(&r);
            });
        }
        for (auto& t : th) t.join();
        return std::all_of(ok.begin(), ok.end(), [](int v) { return v != 0; });
    };
    int warm = (n < 4) ? 30 : 2;
    for (int i = 0; i < warm; i++) if (!run()) return 1;
    std::vector<double> ms;
    for (int i = 0; i < a.runs; i++) {
        double t0 = now_ms();
        if (!run()) return 1;
        ms.push_back(now_ms() - t0);
    }
    if (!a.dump.empty()) {
        FILE* f = std::fopen(a.dump.c_str(), "wb");
        for (auto& v : out) std::fwrite(v.data(), sizeof(float), v.size(), f);
        std::fclose(f);
    }
    report(a, "lib", ms, n, total_tokens, 0, "");
    for (auto* p : ctx) lembed_text_embedding_free(p);
    return 0;
}

/* ------------------------------------------------------------------ raw */
struct RawBatch {
    int bsz, seq;
    std::vector<int64_t> ids, mask, type;
};

std::vector<RawBatch> make_batches(const lembed::detail::TokenizerWrapper& tk,
                                   const std::vector<std::string>& texts, int batch,
                                   long& total, long& padded) {
    std::vector<std::vector<int>> enc;
    for (auto& s : texts) enc.push_back(tk.encode(s));
    std::vector<int> idx(texts.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int x, int y) { return enc[x].size() > enc[y].size(); });
    std::vector<RawBatch> out;
    total = padded = 0;
    for (size_t s = 0; s < idx.size(); s += batch) {
        size_t e = std::min(idx.size(), s + (size_t)batch);
        RawBatch b;
        b.bsz = (int)(e - s);
        b.seq = (int)enc[idx[s]].size();
        b.ids.assign((size_t)b.bsz * b.seq, 0);
        b.mask.assign((size_t)b.bsz * b.seq, 0);
        b.type.assign((size_t)b.bsz * b.seq, 0);
        for (int i = 0; i < b.bsz; i++) {
            const auto& t = enc[idx[s + i]];
            total += (long)t.size();
            for (size_t j = 0; j < t.size(); j++) {
                b.ids[(size_t)i * b.seq + j] = t[j];
                b.mask[(size_t)i * b.seq + j] = 1;
            }
        }
        padded += (long)b.bsz * b.seq;
        out.push_back(std::move(b));
    }
    return out;
}

/* One bare ORT session. Run() only: no output copy, no pooling. */
struct RawSession {
    OrtSession* s = nullptr;
    OrtMemoryInfo* mem = nullptr;
    bool open(const std::string& path, int threads, bool nospin) {
        const OrtApi* api = lembed::detail::ort_api();
        OrtSessionOptions* opts = nullptr;
        lembed::detail::ort_check(api->CreateSessionOptions(&opts));
        lembed::detail::ort_check(api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));
        if (threads > 0) {
            lembed::detail::ort_check(api->SetIntraOpNumThreads(opts, threads));
            lembed::detail::ort_check(api->SetInterOpNumThreads(opts, 1));
        }
        if (nospin)
            lembed::detail::ort_check(api->AddSessionConfigEntry(opts, "session.intra_op.allow_spinning", "0"));
        std::wstring w(path.begin(), path.end());
        lembed::detail::ort_check(api->CreateSession(lembed::detail::ort_env(), w.c_str(), opts, &s));
        api->ReleaseSessionOptions(opts);
        lembed::detail::ort_check(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem));
        return true;
    }
    void run(RawBatch& b) {
        const OrtApi* api = lembed::detail::ort_api();
        int64_t shape[2] = {b.bsz, b.seq};
        size_t bytes = (size_t)b.bsz * b.seq * sizeof(int64_t);
        OrtValue* in[3] = {nullptr, nullptr, nullptr};
        int64_t* data[3] = {b.ids.data(), b.mask.data(), b.type.data()};
        for (int i = 0; i < 3; i++)
            lembed::detail::ort_check(api->CreateTensorWithDataAsOrtValue(
                mem, data[i], bytes, shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in[i]));
        const char* names[] = {"input_ids", "attention_mask", "token_type_ids"};
        const char* outn[] = {"last_hidden_state"};
        OrtValue* outv = nullptr;
        lembed::detail::ort_check(api->Run(s, nullptr, names, in, 3, outn, 1, &outv));
        api->ReleaseValue(outv);
        for (auto* v : in) api->ReleaseValue(v);
    }
};

int run_raw(const Args& a, const std::vector<std::string>& texts) {
    std::string dir = a.model_dir.empty() ? default_model_dir() : a.model_dir;
    lembed::detail::TokenizerWrapper tk;
    tk.load_from_file(dir + "/tokenizer.json", 512);
    const int n = (int)texts.size();
    int per = n / a.workers;
    std::vector<std::vector<RawBatch>> work(a.workers);
    long total = 0, padded = 0;
    for (int w = 0; w < a.workers; w++) {
        int start = w * per, cnt = (w == a.workers - 1) ? n - start : per;
        std::vector<std::string> part(texts.begin() + start, texts.begin() + start + cnt);
        long t, p;
        work[w] = make_batches(tk, part, a.batch, t, p);
        total += t; padded += p;
    }
    std::vector<RawSession> sess(a.workers);
    for (auto& s : sess) s.open(dir + "/model.onnx", a.threads, a.nospin);
    auto run = [&]() {
        std::vector<std::thread> th;
        for (int w = 0; w < a.workers; w++)
            th.emplace_back([&, w]() { for (auto& b : work[w]) sess[w].run(b); });
        for (auto& t : th) t.join();
    };
    int warm = (n < 4) ? 30 : 2;
    for (int i = 0; i < warm; i++) run();
    std::vector<double> ms;
    for (int i = 0; i < a.runs; i++) {
        double t0 = now_ms();
        run();
        ms.push_back(now_ms() - t0);
    }
    report(a, "raw", ms, n, total, padded, "");
    return 0;
}

/* --------------------------------------------------------------- stages */
int run_stages(const Args& a, const std::vector<std::string>& texts) {
    std::string dir = a.model_dir.empty() ? default_model_dir() : a.model_dir;
    lembed::detail::TokenizerWrapper tk;
    tk.load_from_file(dir + "/tokenizer.json", 512);
    lembed::detail::OnnxSession session;
    session.load_from_file((dir + "/model.onnx").c_str(), a.threads, 0, 0);
    const int n = (int)texts.size();
    const int dim = 384;
    int oi = session.select_output();

    std::vector<double> t_count, t_tok, t_tens, t_run, t_pool, t_all;
    for (int rep = 0; rep < a.runs + 2; rep++) {
        double c0 = now_ms();
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::vector<int> tc(n);
        for (int i = 0; i < n; i++) tc[i] = (int)tk.encode(texts[i].c_str()).size();
        std::sort(idx.begin(), idx.end(), [&](int x, int y) { return tc[x] > tc[y]; });
        double c1 = now_ms();
        double tok = 0, tens = 0, run = 0, pool = 0;
        std::vector<float> res((size_t)n * dim);
        int nb = lembed::detail::batch_count(n, a.batch);
        std::vector<int64_t> ids, mask, type;
        std::vector<float> pooled;
        for (int bi = 0; bi < nb; bi++) {
            auto r = lembed::detail::get_batch(bi, n, a.batch);
            int bsz = r.end - r.start;
            double s0 = now_ms();
            std::vector<std::string> bt;
            for (int i = r.start; i < r.end; i++) bt.push_back(texts[idx[i]]);
            auto enc = tk.encode_batch(bt);
            double s1 = now_ms();
            int seq = enc.seq_length;
            size_t fl = (size_t)bsz * seq;
            ids.resize(fl); mask.resize(fl); type.assign(fl, 0);
            for (int i = 0; i < bsz; i++)
                for (int j = 0; j < seq; j++) {
                    ids[i * seq + j] = enc.input_ids[i][j];
                    mask[i * seq + j] = enc.attention_mask[i][j];
                    if ((int)enc.token_type_ids[i].size() > j) type[i * seq + j] = enc.token_type_ids[i][j];
                }
            double s2 = now_ms();
            auto outs = session.run(ids.data(), mask.data(), type.data(), bsz, seq, oi);
            double s3 = now_ms();
            auto& o = outs[0];
            pooled.resize((size_t)bsz * dim);
            lembed::detail::pool_mean(o.data.data(), mask.data(), (int)o.shape[0], (int)o.shape[1],
                                      (int)o.shape[2], (int)o.shape.size(), pooled.data());
            lembed::detail::l2_normalize(pooled.data(), bsz, dim);
            for (int i = 0; i < bsz; i++)
                std::memcpy(res.data() + (size_t)idx[r.start + i] * dim, pooled.data() + (size_t)i * dim,
                            dim * sizeof(float));
            double s4 = now_ms();
            tok += s1 - s0; tens += s2 - s1; run += s3 - s2; pool += s4 - s3;
        }
        double e = now_ms();
        if (rep < 2) continue;
        t_count.push_back(c1 - c0); t_tok.push_back(tok); t_tens.push_back(tens);
        t_run.push_back(run); t_pool.push_back(pool); t_all.push_back(e - c0);
    }
    char extra[512];
    std::snprintf(extra, sizeof extra,
                  ",\"stage_ms\":{\"bucket_count_tokenise\":%.2f,\"tokenise_batch\":%.2f,"
                  "\"tensors\":%.2f,\"ort_run_incl_copy\":%.2f,\"pool_norm_copy\":%.2f}",
                  median(t_count), median(t_tok), median(t_tens), median(t_run), median(t_pool));
    long total = 0;
    for (auto& s : texts) total += (long)tk.encode(s).size();
    report(a, "stages", t_all, n, total, 0, extra);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        auto is = [&](const char* k) { return !std::strcmp(argv[i], k) && i + 1 < argc; };
        if (is("--mode")) a.mode = argv[++i];
        else if (is("--corpus")) a.corpus = argv[++i];
        else if (is("--model-dir")) a.model_dir = argv[++i];
        else if (is("--dump")) a.dump = argv[++i];
        else if (is("--workers")) a.workers = std::max(1, std::atoi(argv[++i]));
        else if (is("--threads")) a.threads = std::atoi(argv[++i]);
        else if (is("--batch")) a.batch = std::max(1, std::atoi(argv[++i]));
        else if (is("--runs")) a.runs = std::max(1, std::atoi(argv[++i]));
        else if (is("--n")) a.n = std::max(1, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--nospin")) a.nospin = true;
    }
    auto texts = make_texts(a.corpus, a.n);
    try {
        if (a.mode == "raw") return run_raw(a, texts);
        if (a.mode == "stages") return run_stages(a, texts);
        return run_lib(a, texts);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
