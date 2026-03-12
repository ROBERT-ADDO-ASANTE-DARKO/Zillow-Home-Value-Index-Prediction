import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/models.dart';
import '../services/api_service.dart';
import 'auth_provider.dart';

// ---------------------------------------------------------------------------
// Selected prime area filter
// ---------------------------------------------------------------------------

/// null = show all prime areas
final selectedPrimeAreaProvider = StateProvider<String?>((ref) => null);

// ---------------------------------------------------------------------------
// Prime area index time series
// ---------------------------------------------------------------------------

final primeIndexProvider =
    FutureProvider.autoDispose<List<LocationPoint>>((ref) async {
  ref.watch(authProvider.select((s) => s.user?.accessToken));
  final area = ref.watch(selectedPrimeAreaProvider);
  final api = ref.watch(apiServiceProvider);
  return api.fetchPrimeIndex(area: area);
});

// ---------------------------------------------------------------------------
// Prime area summary
// ---------------------------------------------------------------------------

final primeSummaryProvider =
    FutureProvider.autoDispose<List<LocationSummary>>((ref) async {
  ref.watch(authProvider.select((s) => s.user?.accessToken));
  final api = ref.watch(apiServiceProvider);
  return api.fetchPrimeSummary();
});
