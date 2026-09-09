/*
 * Production Benchmark - Real-world multilingual corpus
 * Tests robustness and performance with diverse text types
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

/* Real-world multilingual corpus with various lengths */
struct TextSample {
    const char* text;
    const char* lang;
    int tokens_approx;
};

static const TextSample g_corpus[] = {
    /* Short texts (< 20 tokens) */
    {"Hello world.", "en", 2},
    {"Bonjour le monde.", "fr", 3},
    {"Hallo Welt.", "de", 2},
    {"Hola mundo.", "es", 2},
    {"Ciao mondo.", "it", 2},
    {"Olá mundo.", "pt", 2},
    {"Привет мир.", "ru", 2},
    {"こんにちは世界。", "ja", 3},
    {"안녕하세요 세계.", "ko", 3},
    {"你好世界。", "zh", 3},
    {"Machine learning transforms data into insights.", "en", 5},
    {"L'intelligence artificielle transforme les données.", "fr", 5},
    {"Künstliche Intelligenz verändert die Welt.", "de", 5},

    /* Medium texts (20-80 tokens) */
    {"The quick brown fox jumps over the lazy dog near the riverbank while the sun sets behind the mountains.", "en", 18},
    {"Le renard brun rapide saute par-dessus le chien paresseux près de la rivière pendant que le soleil se couche.", "fr", 20},
    {"Der schnelle braune Fuchs springt über den faulen Hund in der Nähe des Flusses, während die Sonne hinter den Bergen untergeht.", "de", 22},
    {"Machine learning algorithms can identify patterns in large datasets automatically without explicit programming instructions.", "en", 14},
    {"Climate change affects global weather patterns and sea levels significantly across all continents and ocean regions worldwide.", "en", 15},
    {"The history of ancient Rome spans over a thousand years of civilization from its founding to the fall of the western empire.", "en", 20},
    {"Quantum computing promises to revolutionize cryptography drug discovery and materials science through parallel processing capabilities.", "en", 14},
    {"Les algorithmes d'apprentissage automatique peuvent identifier des motifs dans de grands ensembles de données.", "fr", 14},
    {"Die künstliche Intelligenz verändert die Art und Weise wie wir arbeiten kommunizieren und Probleme lösen.", "de", 16},
    {"El aprendizaje automático permite a las computadoras aprender de los datos y mejorar con la experiencia.", "es", 15},

    /* Long texts (80-200 tokens) */
    {"Natural language processing is a subfield of linguistics computer science and artificial intelligence concerned with the interactions between computers and human language in particular how to program computers to process and analyze large amounts of natural language data.", "en", 42},
    {"The transformer architecture introduced in the attention is all you need paper has become the foundation for most modern natural language processing systems including BERT GPT and their variants.", "en", 32},
    {"Deep learning is part of a broader family of machine learning methods based on artificial networks with representation learning and has been applied to fields including computer vision speech recognition natural language processing and bioinformatics.", "en", 38},
    {"Le traitement automatique du langage naturel est un domaine de l'informatique et de l'intelligence artificielle qui s'intéresse aux interactions entre les ordinateurs et le langage humain.", "fr", 30},
    {"Die künstliche Intelligenz ist ein Gebiet der Informatik das sich mit der Automatisierung intelligentem Verhalten und dem maschinellen Lernen befasst.", "de", 22},
    {"El procesamiento del lenguaje natural es un campo de la informática la inteligencia artificial y la lingüística interesado en las interacciones entre las computadoras y el lenguaje humano.", "es", 32},

    /* Very long texts (200+ tokens) */
    {"Artificial intelligence has made significant progress in recent years particularly in the areas of machine learning deep learning and natural language processing. These advances have enabled the development of systems that can understand generate and translate human language with remarkable accuracy. Large language models trained on vast amounts of text data have demonstrated capabilities that were previously thought to be decades away including reasoning summarization and creative writing.", "en", 72},
    {"The development of modern artificial intelligence began in the nineteen fifties with the work of Alan Turing and other pioneers who asked whether machines could think. Since then the field has gone through periods of optimism and disappointment known as AI winters. Today we are in a period of rapid advancement driven by increases in computational power the availability of large datasets and improvements in algorithms particularly deep learning.", "en", 78},
    {"L'intelligence artificielle a fait des progrès significatifs ces dernières années en particulier dans les domaines de l'apprentissage automatique de l'apprentissage profond et du traitement du langage naturel. Ces avancées ont permis le développement de systèmes capables de comprendre de générer et de traduire le langage humain avec une précision remarquable.", "fr", 62},
};

static const int g_corpus_size = sizeof(g_corpus) / sizeof(g_corpus[0]);

void embed_task(lembed_text_embedding_t* emb, const char** texts, int n) {
    lembed_embeddings_t res = {0};
    lembed_status_t s = lembed_text_embedding_embed(emb, texts, n, n, &res);
    if (s != LEMBED_OK) {
        fprintf(stderr, "  ERREUR embed: %s\n", lembed_last_error());
    }
    lembed_embeddings_free(&res);
}

int main() {
    printf("=== Production Benchmark - Real Multilingual Corpus ===\n\n");
    printf("Corpus: %d textes, %d langues, longueurs varies\n\n", g_corpus_size, 8);

    /* Test all models */
    struct Test { const char* name; lembed_text_model_t model; };
    Test tests[] = {
        {"MiniLM-L6-v2-Q", LEMBED_TEXT_ALL_MINILM_L6_V2_Q},
        {"MiniLM-L6-v2  ", LEMBED_TEXT_ALL_MINILM_L6_V2},
        {"BGE-small-en  ", LEMBED_TEXT_BGE_SMALL_EN_V15},
    };
    int ntests = sizeof(tests) / sizeof(tests[0]);

    /* Prepare C-strings */
    std::vector<const char*> texts;
    for (int i = 0; i < g_corpus_size; i++)
        texts.push_back(g_corpus[i].text);

    for (int ti = 0; ti < ntests; ti++) {
        printf("--- %s ---\n", tests[ti].name);

        /* Create single session for simplicity */
        lembed_text_options_t opts = lembed_text_options_default();
        opts.model = tests[ti].model;
        opts.num_threads = 4;
        opts.offline = 1;
        opts.show_download_progress = 0;

        lembed_text_embedding_t* emb = nullptr;
        lembed_status_t s = lembed_text_embedding_create(&opts, &emb);
        if (s != LEMBED_OK) {
            printf("  SKIP (erreur chargement: %s)\n\n", lembed_last_error());
            continue;
        }

        int dim = lembed_text_embedding_dim(emb);
        printf("  Dimension: %d\n", dim);

        /* Warmup */
        embed_task(emb, texts.data(), (int)texts.size());

        /* Benchmark single batch */
        double t0 = now_ms();
        embed_task(emb, texts.data(), (int)texts.size());
        double t1 = now_ms();
        printf("  1 batch (%d texts): %.1f ms = %.0f docs/s\n",
               (int)texts.size(), t1-t0, (double)texts.size()/((t1-t0)/1000.0));

        /* Test individual texts to check for failures */
        int failures = 0;
        double total_time = 0;
        for (int i = 0; i < (int)texts.size(); i++) {
            double it0 = now_ms();
            embed_task(emb, texts.data() + i, 1);
            double it1 = now_ms();
            total_time += (it1 - it0);
        }
        printf("  %d textes individuels: %.1f ms total, %.1f ms moy\n",
               (int)texts.size(), total_time, total_time / texts.size());

        /* Verify output validity */
        {
            lembed_embeddings_t res = {0};
            lembed_text_embedding_embed(emb, texts.data(), (int)texts.size(), (int)texts.size(), &res);
            if (res.num_embeddings == (int)texts.size()) {
                /* Check for NaN/Inf */
                int nan_count = 0;
                for (int i = 0; i < res.num_embeddings * res.dim; i++) {
                    if (res.data[i] != res.data[i] || res.data[i] > 1e10 || res.data[i] < -1e10)
                        nan_count++;
                }
                printf("  Validation: %d embeddings, %d valeurs invalides\n",
                       res.num_embeddings, nan_count);
            } else {
                printf("  ERREUR: attendu %d, obtenu %d\n", (int)texts.size(), res.num_embeddings);
                failures++;
            }
            lembed_embeddings_free(&res);
        }

        lembed_text_embedding_free(emb);
        printf("\n");
    }

    printf("=== Termine ===\n");
    return 0;
}
