import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/providers.dart';
import '../widgets/ahpi_chart.dart';
import '../widgets/market_kpis.dart';

class OverviewScreen extends ConsumerWidget {
  const OverviewScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final indexAsync = ref.watch(ahpiIndexProvider);
    final summaryAsync = ref.watch(ahpiSummaryProvider);

    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // KPI cards
          summaryAsync.when(
            data: (summary) => MarketKpis(summary: summary),
            loading: () => const _LoadingCard(height: 100),
            error: (e, _) => _ErrorCard(message: e.toString()),
          ),
          const SizedBox(height: 24),

          // AHPI chart
          _SectionHeader(
            title: 'Composite AHPI',
            subtitle: 'Monthly index (base 2015 = 100)',
          ),
          const SizedBox(height: 12),
          SizedBox(
            height: 340,
            child: indexAsync.when(
              data: (data) => AhpiChart(data: data),
              loading: () => const _LoadingCard(height: 340),
              error: (e, _) => _ErrorCard(message: e.toString()),
            ),
          ),
          const SizedBox(height: 24),

          // Macro regressor selector
          _MacroSection(),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Macro regressor sub-section
// ---------------------------------------------------------------------------

class _MacroSection extends ConsumerWidget {
  const _MacroSection();

  static const _regressors = [
    ('exchange_rate_ghs_usd', 'Exchange Rate'),
    ('cpi_index', 'CPI Index'),
    ('inflation_cpi_pct', 'Inflation %'),
    ('gold_price_usd', 'Gold Price'),
    ('cocoa_price_usd', 'Cocoa Price'),
    ('oil_brent_usd', 'Brent Oil'),
    ('broad_money_pct_gdp', 'Broad Money M2'),
    ('urban_pop_pct', 'Urban Pop %'),
  ];

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final selected = ref.watch(selectedRegressorProvider);
    final macroAsync = ref.watch(macroDataProvider);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _SectionHeader(
          title: 'Macro Regressors',
          subtitle: 'Top Prophet drivers of the AHPI',
        ),
        const SizedBox(height: 8),
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Row(
            children: _regressors.map((r) {
              final isSelected = selected == r.$1;
              return Padding(
                padding: const EdgeInsets.only(right: 8),
                child: ChoiceChip(
                  label: Text(r.$2),
                  selected: isSelected,
                  onSelected: (_) =>
                      ref.read(selectedRegressorProvider.notifier).state = r.$1,
                  selectedColor: const Color(0xFF1C6B2A),
                  labelStyle: TextStyle(
                    color: isSelected ? Colors.white : Colors.grey,
                    fontSize: 12,
                  ),
                  backgroundColor: const Color(0xFF161B22),
                  side: const BorderSide(color: Color(0xFF30363D)),
                ),
              );
            }).toList(),
          ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          height: 260,
          child: macroAsync.when(
            data: (data) => AhpiChart(data: data),
            loading: () => const _LoadingCard(height: 260),
            error: (e, _) => _ErrorCard(message: e.toString()),
          ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

class _SectionHeader extends StatelessWidget {
  const _SectionHeader({required this.title, required this.subtitle});

  final String title;
  final String subtitle;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title,
            style: const TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.w600)),
        Text(subtitle,
            style: const TextStyle(color: Colors.grey, fontSize: 12)),
      ],
    );
  }
}

class _LoadingCard extends StatelessWidget {
  const _LoadingCard({required this.height});

  final double height;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: height,
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(12),
      ),
      child: const Center(
        child: CircularProgressIndicator(color: Color(0xFF4CAF50)),
      ),
    );
  }
}

class _ErrorCard extends StatelessWidget {
  const _ErrorCard({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.red.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.red.withOpacity(0.3)),
      ),
      child: Text(
        'Error: $message',
        style: const TextStyle(color: Colors.redAccent, fontSize: 13),
      ),
    );
  }
}
