import 'package:flutter/foundation.dart';

import '../models/app_models.dart';
import '../services/api_service.dart';

enum LoadingState { idle, loading, loaded, error }

class AppProvider extends ChangeNotifier {
  // ── Selection state ────────────────────────────────────────────────────────
  List<String> cities = [];
  List<String> zipcodes = [];
  String? selectedCity;
  String? selectedZipcode;
  int years = 5;

  // ── Data state ─────────────────────────────────────────────────────────────
  MarketMetrics? metrics;
  List<PricePoint> cityHistory = [];
  List<PricePoint> zipcodeHistory = [];
  ForecastResult? forecast;
  Map<String, double>? cityCoordinates;
  List<CityMarker> cityMarkers = [];

  // ── Event analysis ─────────────────────────────────────────────────────────
  String? eventName;
  DateTime? eventDate;
  int eventImpact = 0;

  // ── Loading / error states ─────────────────────────────────────────────────
  LoadingState citiesState = LoadingState.idle;
  LoadingState dataState = LoadingState.idle;
  LoadingState forecastState = LoadingState.idle;
  String? errorMessage;

  AppProvider() {
    _loadCities();
  }

  // ── Public actions ─────────────────────────────────────────────────────────

  Future<void> selectCity(String city) async {
    if (city == selectedCity) return;
    selectedCity = city;
    zipcodes = [];
    selectedZipcode = null;
    metrics = null;
    cityHistory = [];
    zipcodeHistory = [];
    forecast = null;
    dataState = LoadingState.loading;
    notifyListeners();

    try {
      zipcodes = await ApiService.getZipcodes(city);
      if (zipcodes.isNotEmpty) {
        selectedZipcode = zipcodes.first;
      }
      await _loadCityData();
    } catch (e) {
      dataState = LoadingState.error;
      errorMessage = e.toString();
      notifyListeners();
    }
  }

  Future<void> selectZipcode(String zipcode) async {
    if (zipcode == selectedZipcode) return;
    selectedZipcode = zipcode;
    await _loadCityData();
  }

  Future<void> setYears(int newYears) async {
    years = newYears;
    await _loadForecast();
  }

  Future<void> applyEvent(String name, DateTime date, int impact) async {
    eventName = name;
    eventDate = date;
    eventImpact = impact;
    await _loadForecast();
  }

  void clearEvent() {
    eventName = null;
    eventDate = null;
    eventImpact = 0;
    _loadForecast();
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  Future<void> _loadCities() async {
    citiesState = LoadingState.loading;
    notifyListeners();
    try {
      cities = await ApiService.getCities();
      citiesState = LoadingState.loaded;
      if (cities.isNotEmpty) {
        await selectCity(cities.first);
      }
    } catch (e) {
      citiesState = LoadingState.error;
      errorMessage = e.toString();
      notifyListeners();
    }
  }

  Future<void> _loadCityData() async {
    if (selectedCity == null || selectedZipcode == null) return;
    dataState = LoadingState.loading;
    notifyListeners();

    try {
      // Fire all independent requests in parallel
      final metricsFuture = ApiService.getMetrics(selectedCity!, selectedZipcode!);
      final historyFuture = ApiService.getPriceHistory(selectedCity!, selectedZipcode!);
      final coordsFuture = ApiService.getCityCoordinates(selectedCity!);
      final markersFuture = ApiService.getCityMarkers(selectedCity!);

      metrics = await metricsFuture;
      final history = await historyFuture;
      cityHistory = history['city'] ?? [];
      zipcodeHistory = history['zipcode'] ?? [];
      cityCoordinates = await coordsFuture;
      cityMarkers = await markersFuture;

      dataState = LoadingState.loaded;
    } catch (e) {
      dataState = LoadingState.error;
      errorMessage = e.toString();
      notifyListeners();
      return;
    }
    notifyListeners();
    await _loadForecast();
  }

  Future<void> _loadForecast() async {
    if (selectedZipcode == null) return;
    forecastState = LoadingState.loading;
    notifyListeners();

    try {
      forecast = await ApiService.getForecast(
        zipcode: selectedZipcode!,
        years: years,
        eventName: (eventName?.isNotEmpty ?? false) ? eventName : null,
        eventDate: eventDate != null
            ? '${eventDate!.year.toString().padLeft(4, '0')}'
                '-${eventDate!.month.toString().padLeft(2, '0')}'
                '-${eventDate!.day.toString().padLeft(2, '0')}'
            : null,
        eventImpact: eventImpact,
      );
      forecastState = LoadingState.loaded;
    } catch (e) {
      forecastState = LoadingState.error;
      errorMessage = e.toString();
    }
    notifyListeners();
  }
}
