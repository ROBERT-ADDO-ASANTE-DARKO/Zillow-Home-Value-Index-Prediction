# AHPI Flutter + FastAPI Application

Mobile and web version of the **Accra Home Price Index (AHPI)** built with Flutter (frontend) and FastAPI (backend).

## Architecture

```
ahpi-app/
├── backend/          FastAPI REST API
│   ├── app/
│   │   ├── main.py          App entry-point & lifespan
│   │   ├── auth.py          Auth0 JWT validation (RS256 / JWKS)
│   │   ├── cache.py         Redis async client
│   │   ├── rate_limiter.py  Sliding-window rate limiter
│   │   ├── config.py        Pydantic settings
│   │   └── routers/
│   │       ├── ahpi.py      Composite AHPI + macro endpoints
│   │       ├── districts.py District price index endpoints
│   │       ├── prime.py     Prime area index endpoints
│   │       └── forecasts.py Prophet forecast endpoints
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/         Flutter app (web + mobile)
│   ├── lib/
│   │   ├── main.dart        Entry-point (ProviderScope)
│   │   ├── app.dart         MaterialApp + auth gate
│   │   ├── config/          App-wide constants & coordinates
│   │   ├── models/          Data models
│   │   ├── services/        ApiService (Dio) + AuthService (Auth0)
│   │   ├── providers/       Riverpod providers
│   │   │   ├── auth_provider.dart
│   │   │   ├── ahpi_provider.dart
│   │   │   ├── districts_provider.dart
│   │   │   ├── prime_provider.dart
│   │   │   └── forecast_provider.dart
│   │   ├── screens/
│   │   │   ├── login_screen.dart
│   │   │   ├── home_screen.dart    (nav shell)
│   │   │   ├── overview_screen.dart
│   │   │   ├── districts_screen.dart
│   │   │   ├── prime_screen.dart
│   │   │   ├── forecast_screen.dart
│   │   │   └── map_screen.dart
│   │   └── widgets/
│   │       ├── ahpi_chart.dart
│   │       ├── forecast_chart.dart
│   │       ├── district_chart.dart
│   │       ├── prime_chart.dart
│   │       ├── market_kpis.dart
│   │       └── location_map.dart
│   ├── pubspec.yaml
│   └── web/
│
└── render.yaml       Render deployment (API + static site + Redis)
```

## Key Technology Choices

| Concern | Solution |
|---------|----------|
| State management | **Riverpod** (`flutter_riverpod ^2.6`) – `FutureProvider`, `StateNotifier`, `StateProvider` |
| Authentication | **Auth0** – PKCE web flow via `auth0_flutter`; RS256 JWT verified by backend using JWKS |
| Rate limiting | **Redis** sliding-window (60 req / 60 s per user sub / IP) via `redis-py` async client |
| Charts | **fl_chart** – line charts with confidence interval bands |
| Maps | **flutter_map** + OpenStreetMap tiles – no API key required |
| HTTP | **Dio** with auth interceptor and 401/429 error handling |
| Deployment | **Render** – Python web service (API) + Static site (Flutter web) + Managed Redis |

## Local Development

### Backend

```bash
cd ahpi-app/backend

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure env vars
cp .env.example .env
# Edit .env with your Auth0 domain, audience, and Redis URL

# Start Redis (Docker)
docker run -d -p 6379:6379 redis:7-alpine

# Run the API
uvicorn app.main:app --reload --port 8000
```

API docs available at http://localhost:8000/docs

### Frontend

```bash
cd ahpi-app/frontend

flutter pub get

# Run on web (with Auth0 + API vars)
flutter run -d chrome \
  --dart-define=AUTH0_DOMAIN=your-tenant.auth0.com \
  --dart-define=AUTH0_CLIENT_ID=YOUR_CLIENT_ID \
  --dart-define=API_BASE_URL=http://localhost:8000/api/v1

# Run on Android / iOS
flutter run \
  --dart-define=AUTH0_DOMAIN=your-tenant.auth0.com \
  --dart-define=AUTH0_CLIENT_ID=YOUR_CLIENT_ID \
  --dart-define=API_BASE_URL=http://localhost:8000/api/v1
```

## Auth0 Setup

1. Create a **Regular Web Application** (for Flutter web) or **Native Application** (for mobile) in Auth0 Dashboard.
2. Add Allowed Callback URLs:
   - Web: `https://ahpi-web.onrender.com/login-callback`, `http://localhost:*`
   - Native: `com.ahpi.flutter://login-callback`
3. Add Allowed Logout URLs matching the above.
4. Create an **API** in Auth0 with audience `https://ahpi-api.onrender.com`.
5. Set environment variables (see `.env.example`).

## Render Deployment

```bash
# From repo root
render blueprint apply --file ahpi-app/render.yaml
```

Set secrets in the Render dashboard:
- `AUTH0_DOMAIN`
- `AUTH0_AUDIENCE` (for backend)
- `AUTH0_CLIENT_ID` (for frontend build)

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Unauthenticated health check |
| GET | `/api/v1/ahpi/index` | Full AHPI time series |
| GET | `/api/v1/ahpi/summary` | Key metrics snapshot |
| GET | `/api/v1/ahpi/macro?regressor=exchange_rate_ghs_usd` | Macro regressor series |
| GET | `/api/v1/districts/index?district=Kasoa` | District AHPI |
| GET | `/api/v1/districts/summary` | District summary table |
| GET | `/api/v1/prime/index?area=East+Legon` | Prime area AHPI |
| GET | `/api/v1/prime/summary` | Prime area summary table |
| GET | `/api/v1/forecasts/ahpi/{scenario}` | Aggregate forecast (bear/base/bull) |
| GET | `/api/v1/forecasts/districts/{district}/{scenario}` | District forecast |
| GET | `/api/v1/forecasts/prime/{area}/{scenario}` | Prime area forecast |

All `/api/v1/*` endpoints require `Authorization: Bearer <Auth0 access token>`.
