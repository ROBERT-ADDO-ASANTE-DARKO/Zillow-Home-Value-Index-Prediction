import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/models.dart';
import '../services/api_service.dart';
import 'auth_provider.dart';

// ---------------------------------------------------------------------------
// Selected scenario
// ---------------------------------------------------------------------------

final selectedScenarioProvider = StateProvider<String>((ref) => 'base');

// ---------------------------------------------------------------------------
// Aggregate AHPI forecast
// ---------------------------------------------------------------------------

final ahpiForecastProvider =
    FutureProvider.autoDispose<List<ForecastPoint>>((ref) async {
  ref.watch(authProvider.select((s) => s.user?.accessToken));
  final scenario = ref.watch(selectedScenarioProvider);
  final api = ref.watch(apiServiceProvider);
  return api.fetchAhpiForecast(scenario);
});

// ---------------------------------------------------------------------------
// District forecast – parameterised by district + scenario
// ---------------------------------------------------------------------------

final districtForecastProvider = FutureProvider.autoDispose
    .family<List<ForecastPoint>, ({String district, String scenario})>(
        (ref, params) async {
  ref.watch(authProvider.select((s) => s.user?.accessToken));
  final api = ref.watch(apiServiceProvider);
  return api.fetchDistrictForecast(params.district, params.scenario);
});

// ---------------------------------------------------------------------------
// Prime area forecast
// ---------------------------------------------------------------------------

final primeForecastProvider = FutureProvider.autoDispose
    .family<List<ForecastPoint>, ({String area, String scenario})>(
        (ref, params) async {
  ref.watch(authProvider.select((s) => s.user?.accessToken));
  final api = ref.watch(apiServiceProvider);
  return api.fetchPrimeForecast(params.area, params.scenario);
});
