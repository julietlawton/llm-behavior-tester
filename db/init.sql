CREATE TABLE experiments (
    id SERIAL PRIMARY KEY,
    dataset JSONB NOT NULL,
    models JSONB NOT NULL,
    evaluator_type VARCHAR NOT NULL CHECK (evaluator_type IN ('structured', 'free')),
    evaluator_config JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    experiment_id INT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    model_id VARCHAR NOT NULL,
    model_config JSONB NOT NULL,
    job_status VARCHAR NOT NULL CHECK (job_status IN ('pending', 'running', 'awaiting_eval', 'completed', 'failed', 'cancelled')),
    user_prompt TEXT NOT NULL,
    system_prompt TEXT,
    target_response TEXT,
    model_response TEXT,
    eval_label VARCHAR,
    eval_justification TEXT,
    usage NUMERIC(12, 4) DEFAULT 0,
    error_log JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);