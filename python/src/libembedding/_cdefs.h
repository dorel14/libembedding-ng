/* Flattened C declarations for cffi â€” derived from libembedding public headers.
 * No preprocessor directives, no C++ constructs.
 * Synced with headers in include/libembedding/ (v1.4.0). */

/* â”€â”€ Error handling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef enum {
    LEMBED_OK = 0,
    LEMBED_ERROR_INVALID_ARGUMENT,
    LEMBED_ERROR_OUT_OF_MEMORY,
    LEMBED_ERROR_ONNX_RUNTIME,
    LEMBED_ERROR_TOKENIZER,
    LEMBED_ERROR_DOWNLOAD,
    LEMBED_ERROR_IO,
    LEMBED_ERROR_MODEL_NOT_FOUND,
    LEMBED_ERROR_UNSUPPORTED,
    LEMBED_ERROR_BATCH_SIZE,
    LEMBED_ERROR_LLAMA,
    LEMBED_ERROR_CACHE_MISS,
} lembed_status_t;

const char* lembed_status_message(lembed_status_t status);
const char* lembed_last_error(void);
const char* lembed_version(void);

/* â”€â”€ Enums â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef enum {
    LEMBED_PROVIDER_CPU = 0,
    LEMBED_PROVIDER_CUDA,
    LEMBED_PROVIDER_COREML,
    LEMBED_PROVIDER_DIRECTML,
    LEMBED_PROVIDER_TENSORRT,
    LEMBED_PROVIDER_LLAMACPP,
} lembed_execution_provider_t;

typedef enum {
    LEMBED_BACKEND_ONNX = 0,
    LEMBED_BACKEND_LLAMACPP,
    LEMBED_BACKEND_AUTO,       /* Auto-detect based on model path/name */
} lembed_backend_t;

typedef enum {
    LEMBED_BATCH_SEQUENTIAL = 0,
    LEMBED_BATCH_FIXED,
    LEMBED_BATCH_LENGTH_BUCKET,
} lembed_batch_strategy_t;

typedef enum {
    LEMBED_POOLING_CLS = 0,
    LEMBED_POOLING_MEAN,
} lembed_pooling_t;

typedef enum {
    LEMBED_QUANTIZATION_NONE = 0,
    LEMBED_QUANTIZATION_STATIC,
    LEMBED_QUANTIZATION_DYNAMIC,
} lembed_quantization_t;

typedef enum {
    LEMBED_TEXT_ALL_MINILM_L6_V2 = 0,
    LEMBED_TEXT_ALL_MINILM_L6_V2_Q,
    LEMBED_TEXT_ALL_MINILM_L12_V2,
    LEMBED_TEXT_ALL_MINILM_L12_V2_Q,
    LEMBED_TEXT_ALL_MPNET_BASE_V2,
    LEMBED_TEXT_BGE_BASE_EN_V15,
    LEMBED_TEXT_BGE_BASE_EN_V15_Q,
    LEMBED_TEXT_BGE_LARGE_EN_V15,
    LEMBED_TEXT_BGE_LARGE_EN_V15_Q,
    LEMBED_TEXT_BGE_SMALL_EN_V15,
    LEMBED_TEXT_BGE_SMALL_EN_V15_Q,
    LEMBED_TEXT_NOMIC_EMBED_TEXT_V1,
    LEMBED_TEXT_NOMIC_EMBED_TEXT_V15,
    LEMBED_TEXT_NOMIC_EMBED_TEXT_V15_Q,
    LEMBED_TEXT_PARAPHRASE_ML_MINILM_L12_V2,
    LEMBED_TEXT_PARAPHRASE_ML_MINILM_L12_V2_Q,
    LEMBED_TEXT_PARAPHRASE_ML_MPNET_BASE_V2,
    LEMBED_TEXT_BGE_SMALL_ZH_V15,
    LEMBED_TEXT_BGE_LARGE_ZH_V15,
    LEMBED_TEXT_BGE_M3,
    LEMBED_TEXT_MODERNBERT_EMBED_LARGE,
    LEMBED_TEXT_MULTILINGUAL_E5_SMALL,
    LEMBED_TEXT_MULTILINGUAL_E5_BASE,
    LEMBED_TEXT_MULTILINGUAL_E5_LARGE,
    LEMBED_TEXT_MXBAI_EMBED_LARGE_V1,
    LEMBED_TEXT_MXBAI_EMBED_LARGE_V1_Q,
    LEMBED_TEXT_GTE_BASE_EN_V15,
    LEMBED_TEXT_GTE_BASE_EN_V15_Q,
    LEMBED_TEXT_GTE_LARGE_EN_V15,
    LEMBED_TEXT_GTE_LARGE_EN_V15_Q,
    LEMBED_TEXT_CLIP_VIT_B32,
    LEMBED_TEXT_JINA_EMBEDDINGS_V2_BASE_CODE,
    LEMBED_TEXT_JINA_EMBEDDINGS_V2_BASE_EN,
    LEMBED_TEXT_EMBEDDING_GEMMA_300M,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_XS,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_XS_Q,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_S,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_S_Q,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M_Q,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M_LONG,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M_LONG_Q,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_L,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_L_Q,
    LEMBED_TEXT_MODEL_COUNT,
} lembed_text_model_t;

typedef enum {
    LEMBED_SPARSE_SPLADE_PP_V1 = 0,
    LEMBED_SPARSE_BGE_M3,
    LEMBED_SPARSE_MODEL_COUNT,
} lembed_sparse_model_t;

typedef enum {
    LEMBED_IMAGE_CLIP_VIT_B32 = 0,
    LEMBED_IMAGE_RESNET50,
    LEMBED_IMAGE_UNICOM_VIT_B16,
    LEMBED_IMAGE_UNICOM_VIT_B32,
    LEMBED_IMAGE_NOMIC_EMBED_VISION_V15,
    LEMBED_IMAGE_CLIP_VIT_B32_QUANTIZED,
    LEMBED_IMAGE_MODEL_COUNT,
} lembed_image_model_t;

typedef enum {
    LEMBED_RERANKER_BGE_BASE = 0,
    LEMBED_RERANKER_BGE_V2_M3,
    LEMBED_RERANKER_JINA_V1_TURBO_EN,
    LEMBED_RERANKER_JINA_V2_BASE_MULTILINGUAL,
    LEMBED_RERANKER_JINA_V1_TURBO_EN_QUANTIZED,
    LEMBED_RERANKER_MODEL_COUNT,
} lembed_reranker_model_t;

/* â”€â”€ Structs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef struct {
    const char* model_name;
    const char* model_code;
    const char* model_file;
    const char* description;
    int         dim;
    int         max_tokens;
    int         pooling;
    int         quantization;
} lembed_model_info_t;

typedef struct {
    const char*                name;
    int                        dimension;
    int                        max_length;
    int                        pooling;
    int                        num_threads;
    int                        batch_size;
    lembed_execution_provider_t provider;
    int                        device_id;
} lembed_model_desc_t;

typedef struct {
    uint64_t texts_embedded;
    uint64_t batches_run;
    double   avg_latency_ms;
} lembed_stats_t;

typedef struct lembed_text_embedding    lembed_text_embedding_t;
typedef struct lembed_sparse_embedding  lembed_sparse_embedding_ctx_t;
typedef struct lembed_image_embedding   lembed_image_embedding_t;
typedef struct lembed_reranker          lembed_reranker_t;

typedef struct {
    float*  data;
    int     num_embeddings;
    int     dim;
} lembed_embeddings_t;

typedef struct {
    int32_t* indices;
    float*   values;
    int      length;
} lembed_sparse_embedding_t;

typedef struct {
    lembed_sparse_embedding_t* items;
    int count;
} lembed_sparse_embeddings_t;

typedef struct {
    int   index;
    float score;
} lembed_rerank_result_t;

typedef struct {
    lembed_rerank_result_t* items;
    int count;
} lembed_rerank_results_t;

typedef struct {
    lembed_text_model_t         model;
    lembed_execution_provider_t provider;
    int                         device_id;
    const char*                 cache_dir;
    int                         max_length;
    int                         num_threads;
    int                         show_download_progress;
    int                         batch_size;
    int                         offline;
    int                         pooling;
    int                         dim;
    int                         llama_n_ctx;
    int                         llama_n_gpu_layers;
    int                         llama_verbose;
    int                         llama_n_batch;
    int                         auto_workers;
    int                         cache_size;
    lembed_backend_t            backend;       /* ONNX, LLAMACPP, or AUTO */
    int                         batch_strategy; /* lembed_batch_strategy_t (ONNX only) */
} lembed_text_options_t;

typedef enum {
    LEMBED_MODE_FAST = 0,
    LEMBED_MODE_BALANCED = 1,
    LEMBED_MODE_QUALITY = 2,
} lembed_embedding_mode_t;

const char* lembed_mode_to_string(lembed_embedding_mode_t mode);
lembed_text_model_t lembed_recommended_model_for_mode(lembed_embedding_mode_t mode);

typedef struct {
    lembed_image_model_t        model;
    lembed_execution_provider_t provider;
    int                         device_id;
    const char*                 cache_dir;
    int                         num_threads;
    int                         show_download_progress;
    int                         batch_size;
    int                         offline;
    int                         dim;
} lembed_image_options_t;

typedef struct {
    lembed_reranker_model_t     model;
    lembed_execution_provider_t provider;
    int                         device_id;
    const char*                 cache_dir;
    int                         max_length;
    int                         num_threads;
    int                         show_download_progress;
    int                         batch_size;
    int                         offline;
    /* Backend selection */
    lembed_backend_t            backend;       /* ONNX, LLAMACPP, or AUTO */
    /* llama.cpp specific options */
    int                         llama_n_ctx;  /* context size (0 = model default) */
    int                         llama_n_gpu_layers; /* GPU layers for llama.cpp (-1 = all, 0 = CPU) */
    int                         llama_verbose; /* 1 = enable llama.cpp logging */
    int                         llama_n_batch; /* max tokens per llama_encode() (0 = model default) */
    int                         auto_workers;   /* 1 = auto-detect optimal workers/sessions */
    int                         cache_size;     /* 0 = disabled, >0 = LRU cache capacity */
} lembed_reranker_options_t;

typedef struct {
    lembed_sparse_model_t       model;
    lembed_execution_provider_t provider;
    int                         device_id;
    const char*                 cache_dir;
    int                         max_length;
    int                         num_threads;
    int                         show_download_progress;
    int                         batch_size;
    int                         offline;
    int                         top_k;        /* max terms to keep (0 = all) */
    float                       min_weight;   /* pruning threshold */
    int                         storage_format; /* 0=dict, 1=CSR, 2=numpy */
} lembed_sparse_options_t;

typedef struct {
    const unsigned char* onnx_data;
    size_t               onnx_data_size;
    const unsigned char* tokenizer_json;
    size_t               tokenizer_json_size;
    const unsigned char* config_json;
    size_t               config_json_size;
    lembed_pooling_t     pooling;
    int                  dim;
    int                  max_length;
} lembed_user_defined_model_t;

/* â”€â”€ Functions: Options defaults â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

lembed_text_options_t    lembed_text_options_default(void);
lembed_sparse_options_t  lembed_sparse_options_default(void);
lembed_image_options_t   lembed_image_options_default(void);
lembed_reranker_options_t lembed_reranker_options_default(void);

/* â”€â”€ Functions: Text embedding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

lembed_status_t lembed_text_embedding_create(const lembed_text_options_t* options, lembed_text_embedding_t** out);
lembed_status_t lembed_text_embedding_create_custom(const lembed_user_defined_model_t* model, lembed_execution_provider_t provider, int num_threads, lembed_text_embedding_t** out);
lembed_status_t lembed_text_embedding_create_from_path(const char* dir_path, const lembed_text_options_t* options, lembed_text_embedding_t** out);
lembed_status_t lembed_text_embedding_create_from_gguf_path(const char* gguf_path, const lembed_text_options_t* options, lembed_text_embedding_t** out);
lembed_status_t lembed_text_embedding_create_from_gguf_model(const char* repo, const char* filename, const lembed_text_options_t* options, lembed_text_embedding_t** out);
lembed_status_t lembed_text_embedding_embed(lembed_text_embedding_t* ctx, const char* const* texts, int num_texts, int batch_size, lembed_embeddings_t* result);
lembed_status_t lembed_text_embedding_embed_stream(lembed_text_embedding_t* ctx, const char* const* texts, int num_texts, int batch_size, void (*callback)(const float* embedding, int dim, void* userdata), void* userdata);
int lembed_text_embedding_dim(const lembed_text_embedding_t* ctx);
const lembed_model_desc_t* lembed_text_embedding_desc(const lembed_text_embedding_t* ctx);
const char* lembed_text_embedding_model_name(const lembed_text_embedding_t* ctx);
int lembed_text_embedding_max_length(const lembed_text_embedding_t* ctx);
void lembed_text_embedding_stats(const lembed_text_embedding_t* ctx, lembed_stats_t* out);
void lembed_text_embedding_free(lembed_text_embedding_t* ctx);

/* â”€â”€ Functions: Sparse text embedding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

lembed_status_t lembed_sparse_text_embedding_create(const lembed_sparse_options_t* options, lembed_sparse_embedding_ctx_t** out);
lembed_status_t lembed_sparse_text_embedding_create_from_path(const char* dir_path, const lembed_sparse_options_t* options, lembed_sparse_embedding_ctx_t** out);
lembed_status_t lembed_sparse_text_embedding_embed(lembed_sparse_embedding_ctx_t* ctx, const char* const* texts, int num_texts, int batch_size, const lembed_sparse_options_t* sparse_opts, lembed_sparse_embeddings_t* result);
const lembed_model_desc_t* lembed_sparse_text_embedding_desc(const lembed_sparse_embedding_ctx_t* ctx);
const char* lembed_sparse_text_embedding_model_name(const lembed_sparse_embedding_ctx_t* ctx);
int lembed_sparse_text_embedding_max_length(const lembed_sparse_embedding_ctx_t* ctx);
void lembed_sparse_text_embedding_stats(const lembed_sparse_embedding_ctx_t* ctx, lembed_stats_t* out);
void lembed_sparse_text_embedding_free(lembed_sparse_embedding_ctx_t* ctx);

/* â”€â”€ Functions: Image embedding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

lembed_status_t lembed_image_embedding_create(const lembed_image_options_t* options, lembed_image_embedding_t** out);
lembed_status_t lembed_image_embedding_create_from_path(const char* dir_path, const lembed_image_options_t* options, lembed_image_embedding_t** out);
lembed_status_t lembed_image_embedding_embed_files(lembed_image_embedding_t* ctx, const char* const* file_paths, int num_images, int batch_size, lembed_embeddings_t* result);
lembed_status_t lembed_image_embedding_embed_bytes(lembed_image_embedding_t* ctx, const unsigned char* const* image_data, const int* image_sizes, int num_images, int batch_size, lembed_embeddings_t* result);
int lembed_image_embedding_dim(const lembed_image_embedding_t* ctx);
const lembed_model_desc_t* lembed_image_embedding_desc(const lembed_image_embedding_t* ctx);
const char* lembed_image_embedding_model_name(const lembed_image_embedding_t* ctx);
int lembed_image_embedding_max_length(const lembed_image_embedding_t* ctx);
void lembed_image_embedding_stats(const lembed_image_embedding_t* ctx, lembed_stats_t* out);
void lembed_image_embedding_free(lembed_image_embedding_t* ctx);

/* â”€â”€ Functions: Reranker â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

lembed_status_t lembed_reranker_create(const lembed_reranker_options_t* options, lembed_reranker_t** out);
lembed_status_t lembed_reranker_create_from_path(const char* path, const lembed_reranker_options_t* options, lembed_reranker_t** out);
lembed_status_t lembed_reranker_create_from_gguf_path(const char* gguf_path, const lembed_reranker_options_t* options, lembed_reranker_t** out);
lembed_status_t lembed_reranker_create_from_gguf_model(const char* repo, const char* filename, const lembed_reranker_options_t* options, lembed_reranker_t** out);
lembed_status_t lembed_reranker_rerank(lembed_reranker_t* ctx, const char* query, const char* const* documents, int num_documents, int batch_size, lembed_rerank_results_t* result);
const lembed_model_desc_t* lembed_reranker_desc(const lembed_reranker_t* ctx);
const char* lembed_reranker_model_name(const lembed_reranker_t* ctx);
int lembed_reranker_max_length(const lembed_reranker_t* ctx);
void lembed_reranker_stats(const lembed_reranker_t* ctx, lembed_stats_t* out);
void lembed_reranker_free(lembed_reranker_t* ctx);

/* â”€â”€ Functions: Model registry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

lembed_status_t lembed_get_text_model_info(lembed_text_model_t model, lembed_model_info_t* out);
lembed_status_t lembed_get_sparse_model_info(lembed_sparse_model_t model, lembed_model_info_t* out);
lembed_status_t lembed_get_image_model_info(lembed_image_model_t model, lembed_model_info_t* out);
lembed_status_t lembed_get_reranker_model_info(lembed_reranker_model_t model, lembed_model_info_t* out);
lembed_status_t lembed_list_text_models(const lembed_model_info_t** out, int* count);
lembed_status_t lembed_list_sparse_models(const lembed_model_info_t** out, int* count);
lembed_status_t lembed_list_image_models(const lembed_model_info_t** out, int* count);
lembed_status_t lembed_list_reranker_models(const lembed_model_info_t** out, int* count);
int lembed_find_text_model_by_code(const char* model_code);
int lembed_find_sparse_model_by_code(const char* model_code);
int lembed_find_reranker_model_by_code(const char* model_code);
int lembed_find_image_model_by_code(const char* model_code);

/* â”€â”€ Functions: Downloader â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

lembed_status_t lembed_ensure_text_model(lembed_text_model_t model, const char* cache_dir, int show_progress, int offline, char** model_dir_out);
lembed_status_t lembed_ensure_sparse_model(lembed_sparse_model_t model, const char* cache_dir, int show_progress, int offline, char** model_dir_out);
lembed_status_t lembed_ensure_image_model(lembed_image_model_t model, const char* cache_dir, int show_progress, int offline, char** model_dir_out);
lembed_status_t lembed_ensure_reranker_model(lembed_reranker_model_t model, const char* cache_dir, int show_progress, int offline, char** model_dir_out);
lembed_status_t lembed_ensure_gguf_model(const char* repo, const char* filename, const char* cache_dir, int show_progress, int offline, char** model_path_out);
void lembed_free_string(char* s);

/* â”€â”€ Functions: Similarity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

float lembed_cosine_similarity(const float* a, const float* b, int dim);
float lembed_dot_product(const float* a, const float* b, int dim);
float lembed_euclidean_distance(const float* a, const float* b, int dim);

/* â”€â”€ Functions: Memory free â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

void lembed_embeddings_free(lembed_embeddings_t* result);
void lembed_sparse_embeddings_free(lembed_sparse_embeddings_t* result);
void lembed_rerank_results_free(lembed_rerank_results_t* result);

/* â”€â”€ Autotuner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef enum {
    LEMBED_AUTOTUNE_QUICK = 0,
    LEMBED_AUTOTUNE_FULL
} lembed_autotune_mode_t;

/* Generic text tuning result */
typedef struct {
    int workers;
    int threads;
    int batch_size;
    float throughput_docs_sec;
    float latency_ms;
    float memory_mb;
} lembed_tuning_result_t;

/* Run auto-tuning for a text embedding model.
 * Returns optimal configuration for current hardware. */
lembed_status_t lembed_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_tuning_result_t* result);

/* Run auto-tuning with custom corpus. */
lembed_status_t lembed_autotune_custom(
    const char* model_name,
    const char* const* texts,
    int n_texts,
    lembed_autotune_mode_t mode,
    lembed_tuning_result_t* result);

/* Clear autotune cache for a model (or all if model_name=NULL) */
void lembed_autotune_clear_cache(const char* model_name);

/* Auto model selection result */
typedef struct {
    const char* model_code;
    const char* model_name;
    int dim;
    int workers;
    int threads;
    int batch_size;
    double throughput_docs_sec;
    double latency_ms;
    double memory_mb;
    double score;
} lembed_model_selection_t;

/* Auto-select best model for hardware and use case.
 * use_case: "speed", "quality", or "balanced" (default) */
lembed_status_t lembed_auto_select_model(
    const char* use_case,
    lembed_model_selection_t* result);

/* â”€â”€ Reranker Auto-Tuner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef struct {
    int threads;
    int batch_size;
    int max_tokens;
    float throughput_docs_sec;
    float latency_ms;
    float memory_mb;
    double p95_latency_ms;
} lembed_reranker_tuning_result_t;

typedef enum {
    LEMBED_PROFILE_INTERACTIVE = 0,
    LEMBED_PROFILE_BALANCED = 1,
    LEMBED_PROFILE_QUALITY = 2
} lembed_reranker_profile_t;

typedef enum {
    LEMBED_OBJECTIVE_LATENCY = 0,
    LEMBED_OBJECTIVE_THROUGHPUT = 1,
    LEMBED_OBJECTIVE_BALANCED = 2,
    LEMBED_OBJECTIVE_MEMORY = 3
} lembed_objective_t;

lembed_status_t lembed_reranker_auto_config_profile(
    const char* model_name,
    lembed_reranker_profile_t profile,
    lembed_reranker_tuning_result_t* result);

lembed_status_t lembed_reranker_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_objective_t objective,
    lembed_reranker_tuning_result_t* result);

lembed_status_t lembed_reranker_autotune_constrained(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_objective_t objective,
    int min_tokens,
    double max_latency_ms,
    lembed_reranker_tuning_result_t* result);

lembed_status_t lembed_reranker_autotune_custom(
    const char* model_name,
    const char* const* texts,
    int n_texts,
    lembed_autotune_mode_t mode,
    lembed_objective_t objective,
    lembed_reranker_tuning_result_t* result);

lembed_status_t lembed_reranker_auto_config(
    const char* model_name,
    double target_latency_ms,
    lembed_objective_t objective,
    lembed_reranker_tuning_result_t* result);

void lembed_reranker_autotune_clear_cache(const char* model_name);

/* â”€â”€ Unified Auto-Tuner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef enum {
    LEMBED_TASK_EMBEDDING = 0,
    LEMBED_TASK_RERANKING = 1,
    LEMBED_TASK_IMAGE = 2,
    LEMBED_TASK_SPARSE = 3
} lembed_task_t;

typedef struct {
    int task;
    int threads;
    int batch_size;
    int workers;
    int max_tokens;
    int top_k;
    float min_weight;
    int storage_format;
    float throughput_docs_sec;
    float latency_ms;
    double p95_latency_ms;
    float memory_mb;
} lembed_unified_tuning_result_t;

lembed_status_t lembed_autotune_unified(
    lembed_task_t task,
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_unified_tuning_result_t* result);

lembed_status_t lembed_autotune_unified_config(
    lembed_task_t task,
    const char* model_name,
    double target_latency_ms,
    lembed_unified_tuning_result_t* result);

void lembed_autotune_unified_clear_cache(lembed_task_t task, const char* model_name);

/* â”€â”€ Sparse Auto-Tuner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef struct {
    int top_k;
    float min_weight;
    int storage_format;
    int threads;
    int batch_size;
    float throughput_docs_sec;
    float latency_ms;
    float memory_mb;
} lembed_sparse_tuning_result_t;

lembed_status_t lembed_sparse_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_sparse_tuning_result_t* result);

/* â”€â”€ Image Auto-Tuner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef struct {
    int threads;
    int batch_size;
    float throughput_docs_sec;
    float latency_ms;
    float memory_mb;
} lembed_image_tuning_result_t;

lembed_status_t lembed_image_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_image_tuning_result_t* result);

/* â”€â”€ llama.cpp backend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

int lembed_llama_backend_available(void);
const char* lembed_llama_version(void);

/* â”€â”€ Worker auto-tuning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef struct {
    int optimal_workers;
    int optimal_threads;
    int physical_cores;
    int logical_cores;
} lembed_worker_config_t;

lembed_worker_config_t lembed_detect_optimal_workers(void);
int lembed_recommended_workers_for_model(const char* model_path);

/* â”€â”€ GGUF model registry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef struct {
    const char* name;
    const char* gguf_url;
    const char* model_code;
    const char* description;
    int         dim;
    int         params_m;
    int         file_size_mb;
    float       quality_mteb;
    int         recommended_sessions;
} lembed_gguf_model_info_t;

lembed_status_t lembed_list_gguf_models(const lembed_gguf_model_info_t** out, int* count);
const lembed_gguf_model_info_t* lembed_find_gguf_model(const char* name);
const lembed_gguf_model_info_t* lembed_default_gguf_model(void);

/* â”€â”€ Embedding cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef struct {
    size_t capacity;
    size_t current_size;
    int ttl_seconds;
} lembed_cache_config_t;

typedef struct lembed_cache_t lembed_cache_t;

lembed_cache_config_t lembed_cache_config_default(void);
lembed_cache_t* lembed_cache_create(const lembed_cache_config_t* config);
void lembed_cache_free(lembed_cache_t* cache);
void lembed_cache_clear(lembed_cache_t* cache);
int lembed_cache_get(lembed_cache_t* cache, const char* text, float** out_vec, int* dim);
void lembed_cache_put(lembed_cache_t* cache, const char* text, const float* vec, int dim);
size_t lembed_cache_size(const lembed_cache_t* cache);
size_t lembed_cache_capacity(const lembed_cache_t* cache);

/* â”€â”€ Unified Benchmark â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef enum {
    LEMBED_CORPUS_SHORT = 0,
    LEMBED_CORPUS_MEDIUM,
    LEMBED_CORPUS_LONG,
    LEMBED_CORPUS_VERY_LONG,
    LEMBED_CORPUS_MIXED,
    LEMBED_CORPUS_MULTILINGUAL,
    LEMBED_CORPUS_EDGE_CASES,
} lembed_corpus_type_t;

typedef struct {
    float quality_weight;
    float throughput_weight;
    float cost_weight;
} lembed_benchmark_weights_t;

typedef struct {
    float quality_min;
    float throughput_min;
    float memory_max_mb;
} lembed_benchmark_constraints_t;

typedef struct {
    char        backend[32];
    int         num_threads;
    int         batch_size;
    int         workers;
} lembed_backend_config_t;

typedef struct {
    float       throughput_docs_sec;
    float       latency_p50_ms;
    float       latency_p95_ms;
    float       load_time_ms;
    float       peak_memory_mb;
    int         dim;
    int         num_texts;
    int         num_errors;
} lembed_benchmark_metrics_t;

typedef struct {
    char        model_name[128];
    char        model_path[512];
    char        backend[32];
    lembed_backend_config_t config;
    lembed_benchmark_metrics_t metrics;
    float       score;
    float       quality_score;
    float       file_size_mb;
    int         pareto_rank;
} lembed_benchmark_result_t;

lembed_status_t lembed_benchmark_run(
    const char* model_path,
    const char* backend,
    lembed_corpus_type_t corpus_type,
    const lembed_backend_config_t* config,
    lembed_benchmark_result_t* result);

lembed_status_t lembed_benchmark_autotune(
    const char* model_path,
    const char* backend,
    lembed_objective_t objective,
    lembed_benchmark_result_t* result);

int lembed_benchmark_compare(
    const char* onnx_path,
    const char* gguf_path,
    lembed_corpus_type_t corpus_type,
    lembed_benchmark_result_t* results);

lembed_status_t lembed_benchmark_get_corpus(
    lembed_corpus_type_t type,
    const char* const** out_texts,
    int* out_count);

/* â”€â”€ Model Selection Autotuner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

lembed_status_t lembed_benchmark_select_model(
    const char* model_dir,
    lembed_objective_t objective,
    const lembed_benchmark_constraints_t* constraints,
    const lembed_benchmark_weights_t* custom_weights,
    lembed_benchmark_result_t* result);

lembed_status_t lembed_benchmark_detect_sessions(
    const char* model_path,
    int max_sessions,
    int* optimal_sessions,
    float* best_throughput);

const char* lembed_benchmark_default_cache_dir(void);

/* â”€â”€ Tuning Cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef struct {
    char        cpu_name[128];
    int         physical_cores;
    int         logical_cores;
    char        os_name[64];
    int         ram_mb;
    char        features[256];
} lembed_hardware_info_t;

typedef struct {
    char        libembedding[32];
    char        llama_cpp[32];
} lembed_software_info_t;

typedef struct {
    char        model_id[128];
    char        quantization[16];
    int         dim;
    int         file_size_bytes;
} lembed_model_fingerprint_t;

typedef struct {
    int         num_sessions;
    int         num_threads;
    int         batch_size;
    float       throughput_docs_sec;
    float       latency_p50_ms;
    float       latency_p95_ms;
} lembed_tune_config_result_t;

/* â”€â”€ Cache Fingerprints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef struct {
    char cpu_name[128];
    int physical_cores;
    int logical_cores;
    char os_name[64];
    int ram_mb;
    char features[256];
} lembed_cache_hardware_info_t;

typedef struct {
    char libembedding[32];
    char llama_cpp[32];
} lembed_cache_software_info_t;

typedef struct {
    char model_id[128];
    char quantization[16];
    int dim;
    int file_size_bytes;
} lembed_cache_model_info_t;

/* â”€â”€ Cache Entry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef struct {
    int         cache_schema_version;
    lembed_cache_hardware_info_t hardware;
    lembed_cache_software_info_t software;
    lembed_cache_model_info_t model;
    char        backend[32];
    int         num_configs;
    lembed_tune_config_result_t configs[16];
    int         best_idx;
} lembed_tune_cache_entry_t;

lembed_status_t lembed_cache_detect_hardware(lembed_cache_hardware_info_t* hw);
lembed_status_t lembed_cache_detect_software(lembed_cache_software_info_t* sw);
lembed_status_t lembed_tune_cache_load(
    const lembed_cache_hardware_info_t* hw,
    const lembed_cache_software_info_t* sw,
    const lembed_cache_model_info_t* model,
    const char* backend,
    lembed_tune_cache_entry_t* entry);
lembed_status_t lembed_tune_cache_save(const lembed_tune_cache_entry_t* entry);
lembed_status_t lembed_tune_cache_clear(void);
const char* lembed_tune_cache_path(void);
void lembed_tune_cache_key(
    const lembed_cache_hardware_info_t* hw,
    const lembed_cache_software_info_t* sw,
    const lembed_cache_model_info_t* model,
    const char* backend,
    char* key_out);
void lembed_tune_cache_add_config(lembed_tune_cache_entry_t* entry,
                                  const lembed_tune_config_result_t* config);
void lembed_tune_cache_set_best(lembed_tune_cache_entry_t* entry, int idx);
