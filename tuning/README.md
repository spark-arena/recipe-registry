# Tuning Configs

Pre-computed Triton fused MoE kernel tuning configurations for DGX Spark.

These configs are automatically synced and mounted when running recipes
that benefit from tuned kernels. Generate your own with:

    sparkrun tune sglang <recipe> -H <host>
    sparkrun tune vllm <recipe> -H <host>

