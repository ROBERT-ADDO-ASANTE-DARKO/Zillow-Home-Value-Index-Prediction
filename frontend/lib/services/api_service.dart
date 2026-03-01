import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/app_models.dart';

/// Base URL for the FastAPI backend.
/// Change to your deployed URL in production.
const String kApiBaseUrl = 'http://localhost:8000';

class ApiService {
  static Future<List<String>> getCities() async {
    final res = await http.get(Uri.parse('$kApiBaseUrl/cities'));
    _assertOk(res, 'cities');
    final data = json.decode(res.body) as Map<String, dynamic>;
    return (data['cities'] as List<dynamic>).cast<String>();
  }

  static Future<List<String>> getZipcodes(String city) async {
    final res = await http.get(
      Uri.parse('$kApiBaseUrl/zipcodes/${Uri.encodeComponent(city)}'),
    );
    _assertOk(res, 'zipcodes');
    final data = json.decode(res.body) as Map<String, dynamic>;
    return (data['zipcodes'] as List<dynamic>).cast<String>();
  }

  static Future<MarketMetrics> getMetrics(String city, String zipcode) async {
    final uri = Uri.parse('$kApiBaseUrl/market-metrics').replace(
      queryParameters: {'city': city, 'zipcode': zipcode},
    );
    final res = await http.get(uri);
    _assertOk(res, 'market-metrics');
    return MarketMetrics.fromJson(
      json.decode(res.body) as Map<String, dynamic>,
    );
  }

  static Future<Map<String, List<PricePoint>>> getPriceHistory(
    String city,
    String zipcode,
  ) async {
    final uri = Uri.parse('$kApiBaseUrl/price-history').replace(
      queryParameters: {'city': city, 'zipcode': zipcode},
    );
    final res = await http.get(uri);
    _assertOk(res, 'price-history');
    final data = json.decode(res.body) as Map<String, dynamic>;
    return {
      'city': (data['city_history'] as List<dynamic>)
          .map((p) => PricePoint.fromJson(p as Map<String, dynamic>))
          .toList(),
      'zipcode': (data['zipcode_history'] as List<dynamic>)
          .map((p) => PricePoint.fromJson(p as Map<String, dynamic>))
          .toList(),
    };
  }

  static Future<ForecastResult> getForecast({
    required String zipcode,
    required int years,
    String? eventName,
    String? eventDate,
    int eventImpact = 0,
  }) async {
    final body = <String, dynamic>{
      'zipcode': zipcode,
      'years': years,
      if (eventName != null && eventName.isNotEmpty) 'event_name': eventName,
      if (eventDate != null) 'event_date': eventDate,
      if (eventName != null && eventName.isNotEmpty) 'event_impact': eventImpact,
    };
    final res = await http.post(
      Uri.parse('$kApiBaseUrl/forecast'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode(body),
    );
    _assertOk(res, 'forecast');
    return ForecastResult.fromJson(
      json.decode(res.body) as Map<String, dynamic>,
    );
  }

  static Future<Map<String, double>?> getCityCoordinates(String city) async {
    final res = await http.get(
      Uri.parse('$kApiBaseUrl/city-coordinates/${Uri.encodeComponent(city)}'),
    );
    if (res.statusCode == 404) return null;
    _assertOk(res, 'city-coordinates');
    final data = json.decode(res.body) as Map<String, dynamic>;
    return {
      'lat': (data['lat'] as num).toDouble(),
      'lon': (data['lon'] as num).toDouble(),
    };
  }

  static Future<List<CityMarker>> getCityMarkers(String city) async {
    final res = await http.get(
      Uri.parse('$kApiBaseUrl/city-markers/${Uri.encodeComponent(city)}'),
    );
    if (res.statusCode != 200) return [];
    final data = json.decode(res.body) as List<dynamic>;
    return data
        .map((m) => CityMarker.fromJson(m as Map<String, dynamic>))
        .toList();
  }

  static void _assertOk(http.Response res, String endpoint) {
    if (res.statusCode != 200) {
      throw Exception('$endpoint returned ${res.statusCode}: ${res.body}');
    }
  }
}
