<p align="center">
<img src="assets/spark-arena-logo.png" alt="Spark Arena" width="120" />
</p>
<h2 align="center">Spark Arena Team Official Inference Recipes for NVIDIA DGX Spark</h2>
<p align="center">
   <a href="https://spark-arena.com"><img src="https://img.shields.io/badge/Spark_Arena-community-76b900" alt="Spark Arena" /></a>  
   <a href="https://github.com/spark-arena/sparkrun"><img src="https://img.shields.io/badge/sparkrun-CLI-1e40af" alt="sparkrun CLI" /></a>
   <a href="https://sparkrun.dev"><img src="https://img.shields.io/badge/docs-sparkrun.dev-1e40af" alt="Documentation" /></a>

[//]: # (  <a href="https://recipes.sparkrun.dev"><img src="https://img.shields.io/badge/browse-recipes-76b900" alt="Browse Recipes" /></a>)
</p>

---

This is the **official recipe registry** for [sparkrun](https://github.com/spark-arena/sparkrun) from
the [Spark Arena](https://spark-arena.com) team.

Community recipes are run with either no prefix or with the `@official` prefix:

```bash
sparkrun run our-awesome-recipe
# -- or --
sparkrun run @official/our-awesome-recipe
```

### Recipe format

Recipes follow the standard sparkrun recipe schema. See
the [recipe authoring docs](https://sparkrun.dev/recipes/format/) for the full specification.

## Run Official Recipes

```bash
# List available recipes
sparkrun list @official

# Run a recipe
sparkrun run our-awesome-recipe

# Check VRAM requirements before launching
sparkrun show @official/our-awesome-recipe
```

## Links

- [Spark Arena](https://spark-arena.com) — community benchmarking hub
- [sparkrun](https://github.com/spark-arena/sparkrun) — the tool that runs recipes
- [sparkrun docs](https://sparkrun.dev) — full documentation

[//]: # (- [Recipe Explorer]&#40;https://recipes.sparkrun.dev&#41; — browse and filter all recipes)

