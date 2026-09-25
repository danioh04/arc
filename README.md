# arc

An NBA prospect evaluation platform.

Live demo at [danioh04.github.io/arc.html](https://danioh04.github.io/arc.html).

## Run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,server,pipeline]"
cp .env.example .env        # (optional) set Redis caching and SQLite database
uvicorn arc.main:app --reload --port 8000 --env-file .env
```

## Endpoints

| Method | Path |
| --- | --- |
| `GET` | `/api/v1/health` |
| `GET` | `/api/v1/models/metadata` |
| `GET` | `/api/v1/prospects` |
| `GET` | `/api/v1/prospects/{id_or_name}` |
| `GET` | `/api/v1/prospects/{id_or_name}/predict` |
| `GET` | `/api/v1/prospects/{id_or_name}/comparisons?k=5` |
| `GET` | `/api/v1/draft-classes` |
| `GET` | `/api/v1/draft-classes/{year}/big-board` |
| `POST` | `/api/v1/simulator` |
