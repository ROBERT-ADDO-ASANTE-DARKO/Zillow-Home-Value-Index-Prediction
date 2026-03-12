import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../config/constants.dart';
import '../providers/providers.dart';
import '../widgets/district_chart.dart';

class DistrictsScreen extends ConsumerWidget {
  const DistrictsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final selectedDistrict = ref.watch(selectedDistrictProvider);
    final indexAsync = ref.watch(districtIndexProvider);
    final summaryAsync = ref.watch(districtSummaryProvider);

    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // District filter chips
          const Text('Districts',
              style: TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: [
                // "All" chip
                Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: ChoiceChip(
                    label: const Text('All'),
                    selected: selectedDistrict == null,
                    onSelected: (_) =>
                        ref.read(selectedDistrictProvider.notifier).state = null,
                    selectedColor: const Color(0xFF1C6B2A),
                    labelStyle: TextStyle(
                      color: selectedDistrict == null ? Colors.white : Colors.grey,
                      fontSize: 12,
                    ),
                    backgroundColor: const Color(0xFF161B22),
                    side: const BorderSide(color: Color(0xFF30363D)),
                  ),
                ),
                ...AppConstants.districts.map((d) {
                  final isSelected = selectedDistrict == d;
                  return Padding(
                    padding: const EdgeInsets.only(right: 8),
                    child: ChoiceChip(
                      label: Text(d),
                      selected: isSelected,
                      onSelected: (_) =>
                          ref.read(selectedDistrictProvider.notifier).state = d,
                      selectedColor: const Color(0xFF1C6B2A),
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
          const Text('District AHPI Time Series',
              style: TextStyle(color: Colors.white70, fontSize: 13)),
          const SizedBox(height: 8),
          SizedBox(
            height: 320,
            child: indexAsync.when(
              data: (data) => DistrictChart(data: data),
              loading: () => const _LoadingCard(height: 320),
              error: (e, _) => _ErrorCard(message: e.toString()),
            ),
          ),
          const SizedBox(height: 24),

          // Summary table
          const Text('District Summary',
              style: TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600)),
          const SizedBox(height: 12),
          summaryAsync.when(
            data: (summaries) => _DistrictTable(summaries: summaries),
            loading: () => const _LoadingCard(height: 200),
            error: (e, _) => _ErrorCard(message: e.toString()),
          ),
        ],
      ),
    );
  }
}

class _DistrictTable extends StatelessWidget {
  const _DistrictTable({required this.summaries});

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
