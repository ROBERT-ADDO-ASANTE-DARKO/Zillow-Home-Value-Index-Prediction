import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/providers.dart';
import '../widgets/location_map.dart';

class MapScreen extends ConsumerWidget {
  const MapScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final districtSummaryAsync = ref.watch(districtSummaryProvider);
    final primeSummaryAsync = ref.watch(primeSummaryProvider);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: const [
              Text('Accra Location Map',
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.w600)),
              Text(
                'Circle size proportional to AHPI · colour = cold (low) → warm (high)',
                style: TextStyle(color: Colors.grey, fontSize: 12),
              ),
            ],
          ),
        ),
        Expanded(
          child: districtSummaryAsync.when(
            data: (districtSummaries) => primeSummaryAsync.when(
              data: (primeSummaries) => LocationMap(
                districtSummaries: districtSummaries,
                primeSummaries: primeSummaries,
              ),
              loading: () => const Center(
                  child: CircularProgressIndicator(color: Color(0xFF4CAF50))),
              error: (e, _) => _ErrorCard(message: e.toString()),
            ),
            loading: () => const Center(
                child: CircularProgressIndicator(color: Color(0xFF4CAF50))),
            error: (e, _) => _ErrorCard(message: e.toString()),
          ),
        ),
      ],
    );
  }
}

class _ErrorCard extends StatelessWidget {
  const _ErrorCard({required this.message});
  final String message;
  @override
  Widget build(BuildContext context) => Center(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Text('Map error: $message',
              style: const TextStyle(color: Colors.redAccent)),
        ),
      );
}
