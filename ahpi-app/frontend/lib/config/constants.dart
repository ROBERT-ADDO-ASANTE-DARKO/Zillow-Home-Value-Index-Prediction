/// Application-wide configuration constants.
///
/// All Auth0 / API values are read from compile-time --dart-define flags
/// so that secrets are never hard-coded in source.
class AppConstants {
  AppConstants._();

  // ---------------------------------------------------------------------------
  // Auth0
  // ---------------------------------------------------------------------------

  /// Auth0 domain – e.g. "your-tenant.auth0.com"
  static const String auth0Domain =
      String.fromEnvironment('AUTH0_DOMAIN', defaultValue: 'your-tenant.auth0.com');

  /// Auth0 client ID of the Flutter SPA / Native app
  static const String auth0ClientId =
      String.fromEnvironment('AUTH0_CLIENT_ID', defaultValue: 'YOUR_CLIENT_ID');

  // ---------------------------------------------------------------------------
  // API
  // ---------------------------------------------------------------------------

  /// Base URL of the FastAPI backend
  static const String apiBaseUrl =
      String.fromEnvironment('API_BASE_URL', defaultValue: 'http://localhost:8000/api/v1');

  // ---------------------------------------------------------------------------
  // Design
  // ---------------------------------------------------------------------------

  static const double borderRadius = 12.0;
  static const double cardElevation = 2.0;
  static const double chartPadding = 16.0;

  // ---------------------------------------------------------------------------
  // Districts & areas
  // ---------------------------------------------------------------------------

  static const List<String> districts = [
    'Spintex Road',
    'Adenta',
    'Tema',
    'Dome',
    'Kasoa',
  ];

  static const List<String> primeAreas = [
    'East Legon',
    'Cantonments',
    'Airport Residential',
    'Labone/Roman Ridge',
    'Dzorwulu/Abelenkpe',
    'Trasacco Valley',
  ];

  static const List<String> scenarios = ['bear', 'base', 'bull'];

  // ---------------------------------------------------------------------------
  // District map centre coordinates [lat, lon]
  // ---------------------------------------------------------------------------

  static const Map<String, List<double>> districtCoordinates = {
    'Spintex Road': [5.6485, -0.1042],
    'Adenta': [5.7060, -0.1652],
    'Tema': [5.6698, -0.0166],
    'Dome': [5.6407, -0.2412],
    'Kasoa': [5.5328, -0.4149],
  };

  static const Map<String, List<double>> primeCoordinates = {
    'East Legon': [5.6333, -0.1500],
    'Cantonments': [5.5831, -0.1757],
    'Airport Residential': [5.6061, -0.1719],
    'Labone/Roman Ridge': [5.5694, -0.1772],
    'Dzorwulu/Abelenkpe': [5.5958, -0.2097],
    'Trasacco Valley': [5.6667, -0.2000],
  };

  // Accra map centre
  static const List<double> accraCenter = [5.6037, -0.1870];
}
