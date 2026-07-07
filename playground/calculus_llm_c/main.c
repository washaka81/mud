#include "include/math_c.h"
#include "include/tokenizer_c.h"
#include <string.h>
#include <time.h>

typedef struct {
    State target;
    double lambda;
    double sigma;
} SDEContext;

State drift_func(State y, double t, void* user_data) {
    SDEContext* ctx = (SDEContext*)user_data;
    return mul(sub(ctx->target, y), ctx->lambda);
}

State diffusion_func(State y, double t, void* user_data) {
    SDEContext* ctx = (SDEContext*)user_data;
    State noise;
    for (int i = 0; i < DIM; i++) {
        // Box-Muller transform para ruido Gaussiano N(0,1)
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double z0 = sqrt(-2.0 * log(u1 + 1e-12)) * cos(2.0 * 3.1415926535 * u2);
        noise.data[i] = z0 * ctx->sigma;
    }
    return noise;
}

int main() {
    srand(time(NULL));
    Tokenizer* tok = create_tokenizer();
    load_vocab_c(tok, NULL);

    printf("========================================\n");
    printf("   LLM Basado en Cálculo (Versión C) \n");
    printf("========================================\n");

    char input[256];
    while (1) {
        printf("Prompt C > ");
        if (!fgets(input, 256, stdin)) break;
        input[strcspn(input, "\n")] = 0;
        if (strcmp(input, "salir") == 0) break;

        SDEContext ctx;
        ctx.lambda = 0.8;
        ctx.sigma = 0.005; // Ajuste del sigma al escalar bien el ruido
        
        // Determinar target simplificado para C (mejora iterativa)
        if (strstr(input, "hola") != NULL) ctx.target = encode_c(tok, "mundo");
        else if (strstr(input, "quien") != NULL) ctx.target = encode_c(tok, "ia");
        else ctx.target = encode_c(tok, "verdad");

        printf("[C-INFO] Convergiendo hacia: %s\n", decode_c(tok, ctx.target));

        // Initialize State: first element = 0.1, rest zero-initialized by C aggregate rules
        State current = {{0.1}};
        double t = 0;
        double dt = 0.1;
        while (t < 5.0) {
            current = euler_maruyama_step(drift_func, diffusion_func, current, t, dt, &ctx);
            t += dt;
        }

        printf("\n[C-LLM] Respuesta: %s\n", decode_c(tok, current));
        printf("----------------------------------------\n");
    }

    free_tokenizer(tok);
    return 0;
}
