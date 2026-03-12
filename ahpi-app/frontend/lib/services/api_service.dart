import 'package:dio/dio.dart';

import '../config/constants.dart';
import '../models/models.dart';

/// HTTP client for the AHPI FastAPI backend.
///
/// All requests are authenticated with the Auth0 access token passed
/// via [setAccessToken]. The token is refreshed by the auth provider
/// whenever it rotates.
class ApiService {
  ApiService() {
    _dio = Dio(
      BaseOptions(
        baseUrl: AppConstants.apiBaseUrl,
        connectTimeout: const Duration(seconds: 10),
        receiveTimeout: const Duration(seconds: 20),
      ),
    );

    // Attach auth header on every request
    _dio.interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) {
          if (_accessToken != null) {
            options.headers['Authorization'] = 'Bearer $_accessToken';
          }
          return handler.next(options);
        },
        onError: (error, handler) {
          // Surface 401 / 429 as readable exceptions
          final code = error.response?.statusCode;
          if (code == 401) throw Exception('Authentication required. Please log in again.');
          if (code == 429) {
            final retryAfter = error.response?.headers.value('Retry-After') ?? '60';
            throw Exception('Rate limit exceeded. Retry after ${retryAfter}s.');
          }
          return handler.next(error);
        },
      ),
    );
  }

  late final Dio _dio;
  String? _accessToken;

  void setAccessToken(String token) => _accessToken = token;
  void clearAccessToken() => _accessToken = null;

  // ---------------------------------------------------------------------------
  // AHPI Overview
  // ---------------------------------------------------------------------------

  Future<List<AhpiPoint>> fetchAhpiIndex() async {
    final response = await _dio.get<Map<String, dynamic>>('/ahpi/index');
    final data = response.data!['data'] as List<dynamic>;
    return data.map((e) => AhpiPoint.fromJson(e as Map<String, dynamic>)).toList();
  }

  Future<AhpiSummary> fetchAhpiSummary() async {
    final response = await _dio.get<Map<String, dynamic>>('/ahpi/summary');
    return AhpiSummary.fromJson(response.data!);
  }

  Future<List<AhpiPoint>> fetchMacroData({String? regressor}) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '/ahpi/macro',
      queryParameters: regressor != null ? {'regressor': regressor} : null,
    );
    final data = response.data!['data'] as List<dynamic>;
    return data.map((e) => AhpiPoint.fromJson(e as Map<String, dynamic>)).toList();
  }

  // ---------------------------------------------------------------------------
  // Districts
  // ---------------------------------------------------------------------------

  Future<List<LocationPoint>> fetchDistrictIndex({String? district}) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '/districts/index',
      queryParameters: district != null ? {'district': district} : null,
    );
    final data = response.data!['data'] as List<dynamic>;
    return data
        .map((e) => LocationPoint.fromJson(e as Map<String, dynamic>, locationKey: 'district'))
        .toList();
  }

  Future<List<LocationSummary>> fetchDistrictSummary() async {
    final response = await _dio.get<Map<String, dynamic>>('/districts/summary');
    final data = response.data!['summary'] as List<dynamic>;
    return data
        .map((e) => LocationSummary.fromJson(e as Map<String, dynamic>, nameKey: 'district'))
        .toList();
  }

  // ---------------------------------------------------------------------------
  // Prime Areas
  // ---------------------------------------------------------------------------

  Future<List<LocationPoint>> fetchPrimeIndex({String? area}) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '/prime/index',
      queryParameters: area != null ? {'area': area} : null,
    );
    final data = response.data!['data'] as List<dynamic>;
    return data
        .map((e) => LocationPoint.fromJson(e as Map<String, dynamic>, locationKey: 'area'))
        .toList();
  }

  Future<List<LocationSummary>> fetchPrimeSummary() async {
    final response = await _dio.get<Map<String, dynamic>>('/prime/summary');
    final data = response.data!['summary'] as List<dynamic>;
    return data
        .map((e) => LocationSummary.fromJson(e as Map<String, dynamic>, nameKey: 'area'))
        .toList();
  }

  // ---------------------------------------------------------------------------
  // Forecasts
  // ---------------------------------------------------------------------------

  Future<List<ForecastPoint>> fetchAhpiForecast(String scenario) async {
    final response = await _dio.get<Map<String, dynamic>>('/forecasts/ahpi/$scenario');
    final data = response.data!['data'] as List<dynamic>;
    return data.map((e) => ForecastPoint.fromJson(e as Map<String, dynamic>)).toList();
  }

  Future<List<ForecastPoint>> fetchDistrictForecast(
    String district,
    String scenario,
  ) async {
    final encodedDistrict = Uri.encodeComponent(district);
    final response =
        await _dio.get<Map<String, dynamic>>('/forecasts/districts/$encodedDistrict/$scenario');
    final data = response.data!['data'] as List<dynamic>;
    return data.map((e) => ForecastPoint.fromJson(e as Map<String, dynamic>)).toList();
  }

  Future<List<ForecastPoint>> fetchPrimeForecast(
    String area,
    String scenario,
  ) async {
    final encodedArea = Uri.encodeComponent(area);
    final response =
        await _dio.get<Map<String, dynamic>>('/forecasts/prime/$encodedArea/$scenario');
    final data = response.data!['data'] as List<dynamic>;
    return data.map((e) => ForecastPoint.fromJson(e as Map<String, dynamic>)).toList();
  }
}
