"""Neural Router experiment package.

Core modules:
  data        -- Dataset loading for D1 (CardiffNLP), D2 (MultiEURLEX), D3 (MN-DS)
  router      -- Neural Router engine (Algorithms 1-3 from the paper)
  llm         -- LLM client abstraction (LiteLLM)
  llm_async   -- Async LLM client for parallel invocations
  embeddings  -- Embedding model with disk caching
  baselines   -- Six baseline matching methods
  evaluation  -- Metrics computation and statistical tests
"""
