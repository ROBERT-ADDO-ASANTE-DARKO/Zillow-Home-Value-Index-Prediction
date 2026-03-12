import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/models.dart';
import '../services/api_service.dart';
import 'auth_provider.dart';

// ---------------------------------------------------------------------------
// AHPI index time series
// ---------------------------------------------------------------------------

final ahpiIndexProvider = FutureProvider.autoDispose<List<AhpiPoint>>((ref) async {
  // Re-fetch when the user's token changes
  ref.watch(authProvider.select((s) => s.user?.accessToken));
  final api = ref.watch(apiServiceProvider);
  return api.fetchAhpiIndex();
});

// ---------------------------------------------------------------------------
// AHPI summary KPIs
// ---------------------------------------------------------------------------

final ahpiSummaryProvider = FutureProvider.autoDispose<AhpiSummary>((ref) async {
  ref.watch(authProvider.select((s) => s.user?.accessToken));
  final api = ref.watch(apiServiceProvider);
  return api.fetchAhpiSummary();
});

// ---------------------------------------------------------------------------
// Macro regressor (single selected column)
// ---------------------------------------------------------------------------

final selectedRegressorProvider = StateProvider<String>((ref) => 'exchange_rate_ghs_usd');

final macroDataProvider = FutureProvider.autoDispose<List<AhpiPoint>>((ref) async {
  ref.watch(authProvider.select((s) => s.user?.accessToken));
  final regressor = ref.watch(selectedRegressorProvider);
  final api = ref.watch(apiServiceProvider);
  return api.fetchMacroData(regressor: regressor);
});
