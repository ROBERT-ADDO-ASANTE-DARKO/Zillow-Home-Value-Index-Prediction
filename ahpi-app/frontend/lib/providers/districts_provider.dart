import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/models.dart';
import '../services/api_service.dart';
import 'auth_provider.dart';

// ---------------------------------------------------------------------------
// Selected district filter
// ---------------------------------------------------------------------------

/// null = show all districts
final selectedDistrictProvider = StateProvider<String?>((ref) => null);

// ---------------------------------------------------------------------------
// District index time series
// ---------------------------------------------------------------------------

final districtIndexProvider =
    FutureProvider.autoDispose<List<LocationPoint>>((ref) async {
  ref.watch(authProvider.select((s) => s.user?.accessToken));
  final district = ref.watch(selectedDistrictProvider);
  final api = ref.watch(apiServiceProvider);
  return api.fetchDistrictIndex(district: district);
});

// ---------------------------------------------------------------------------
// District summary (latest values + % change per district)
// ---------------------------------------------------------------------------

final districtSummaryProvider =
    FutureProvider.autoDispose<List<LocationSummary>>((ref) async {
  ref.watch(authProvider.select((s) => s.user?.accessToken));
  final api = ref.watch(apiServiceProvider);
  return api.fetchDistrictSummary();
});
