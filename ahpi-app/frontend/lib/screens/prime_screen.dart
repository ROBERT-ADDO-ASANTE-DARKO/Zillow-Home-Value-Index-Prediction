import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../config/constants.dart';
import '../providers/providers.dart';
import '../widgets/prime_chart.dart';

class PrimeScreen extends ConsumerWidget {
  const PrimeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final selectedArea = ref.watch(selectedPrimeAreaProvider);
    final indexAsync = ref.watch(primeIndexProvider);
    final summaryAsync = ref.watch(primeSummaryProvider);

    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Prime Areas',
              style: TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600)),
          const SizedBox(height: 4),
          const Text(
            'USD-indexed luxury market · average 1.8× mid-market composite',
            style: TextStyle(color: Colors.grey, fontSize: 12),
          ),
          const SizedBox(height: 12),

          // Area filter chips
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: [
                Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: ChoiceChip(
                    label: const Text('All'),
                    selected: selectedArea == null,
                    onSelected: (_) =>
                        ref.read(selectedPrimeAreaProvider.notifier).state = null,
                    selectedColor: const Color(0xFF7B1FA2),
                    labelStyle: TextStyle(
                      color: selectedArea == null ? Colors.white : Colors.grey,
                      fontSize: 12,
                    ),
                    backgroundColor: const Color(0xFF161B22),
                    side: const BorderSide(color: Color(0xFF30363D)),
                  ),
                ),
                ...AppConstants.primeAreas.map((a) {
                  final isSelected = selectedArea == a;
                  return Padding(
                    padding: const EdgeInsets.only(right: 8),
                    child: ChoiceChip(
                      label: Text(a),
                      selected: isSelected,
                      onSelected: (_) =>
                          ref.read(selectedPrimeAreaProvider.notifier).state = a,
                      selectedColor: const Color(0xFF7B1FA2),
                      labelStyle: TextStyle(
                        color: isSelected ? Colors.white : Colors.grey,
                        fontSize: 12,
                      ),
                      backgroundColor: const Color(0xFF161B22),
                      side: const BorderSide(color: Color(0xFF30363D)),
                    ),
                  );
                }),
              ],
            ),
          ),
          const SizedBox(height: 20),

          // Chart
          const Text('Prime Area AHPI Time Series',
              style: TextStyle(color: Colors.white70, fontSize: 13)),
          const SizedBox(height: 8),
          SizedBox(
            height: 320,
            child: indexAsync.when(
              data: (data) => PrimeChart(data: data),
              loading: () => const _LoadingCard(height: 320),
              error: (e, _) => _ErrorCard(message: e.toString()),
            ),
          ),
          const SizedBox(height: 24),

          // Summary table
          const Text('Prime Area Summary',
              style: TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600)),
          const SizedBox(height: 12),
          summaryAsync.when(
            data: (summaries) => _PrimeTable(summaries: summaries),
            loading: () => const _LoadingCard(height: 200),
            error: (e, _) => _ErrorCard(message: e.toString()),
          ),
        ],
      ),
    );
  }
}

class _PrimeTable extends StatelessWidget {
  const _PrimeTable({required this.summaries});
  final List summaries;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: summaries.map<Widget>((s) {
        final isPositive = s.pctChange >= 0;
        return Container(
          margin: const EdgeInsets.only(bottom: 8),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          decoration: BoxDecoration(
            color: const Color(0xFF161B22),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: const Color(0xFF30363D)),
          ),
          child: Row(
            children: [
              Expanded(
                flex: 3,
                child: Text(s.name,
                    style: const TextStyle(
                        color: Colors.white, fontWeight: FontWeight.w500)),
              ),
              Expanded(
                flex: 2,
                child: Text(
                  s.latestAhpi.toStringAsFixed(1),
                  style: const TextStyle(color: Colors.white70),
                  textAlign: TextAlign.center,
                ),
              ),
              Expanded(
                flex: 2,
                child: Text(
                  '${isPositive ? '+' : ''}${s.pctChange.toStringAsFixed(1)}%',
                  style: TextStyle(
                    color: isPositive ? Colors.greenAccent : Colors.redAccent,
                    fontWeight: FontWeight.w600,
                  ),
                  textAlign: TextAlign.right,
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }
}

class _LoadingCard extends StatelessWidget {
  const _LoadingCard({required this.height});
  final double height;
  @override
  Widget build(BuildContext context) => Container(
        height: height,
        decoration: BoxDecoration(
          color: const Color(0xFF161B22),
          borderRadius: BorderRadius.circular(12),
        ),
        child: const Center(
            child: CircularProgressIndicator(color: Color(0xFF4CAF50))),
      );
}

class _ErrorCard extends StatelessWidget {
  const _ErrorCard({required this.message});
  final String message;
  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.red.withOpacity(0.1),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.red.withOpacity(0.3)),
        ),
        child: Text('Error: $message',
            style: const TextStyle(color: Colors.redAccent, fontSize: 13)),
      );
}
