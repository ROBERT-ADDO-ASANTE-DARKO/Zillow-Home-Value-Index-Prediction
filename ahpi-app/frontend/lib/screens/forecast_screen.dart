import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/providers.dart';
import '../widgets/forecast_chart.dart';

class ForecastScreen extends ConsumerWidget {
  const ForecastScreen({super.key});

  static const _scenarioDescriptions = {
    'bear': 'Continued cedi depreciation · high inflation · lower commodity prices',
    'base': 'Gradual stabilisation · mid cocoa revenues · moderate inflation',
    'bull': 'Cedi recovery on cocoa windfall · gold rally · falling inflation',
  };

  static const _scenarioColors = {
    'bear': Color(0xFFE57373),
    'base': Color(0xFF4CAF50),
    'bull': Color(0xFF64B5F6),
  };

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final scenario = ref.watch(selectedScenarioProvider);
    final historicalAsync = ref.watch(ahpiIndexProvider);
    final forecastAsync = ref.watch(ahpiForecastProvider);

    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Prophet Forecasts',
              style: TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600)),
          const Text(
            'Jan 2025 – Dec 2026 · 3 economic scenarios',
            style: TextStyle(color: Colors.grey, fontSize: 12),
          ),
          const SizedBox(height: 16),

          // Scenario selector
          Row(
            children: ['bear', 'base', 'bull'].map((s) {
              final isSelected = scenario == s;
              final color = _scenarioColors[s]!;
              return Padding(
                padding: const EdgeInsets.only(right: 8),
                child: GestureDetector(
                  onTap: () =>
                      ref.read(selectedScenarioProvider.notifier).state = s,
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 10),
                    decoration: BoxDecoration(
                      color: isSelected
                          ? color.withOpacity(0.2)
                          : const Color(0xFF161B22),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(
                        color: isSelected ? color : const Color(0xFF30363D),
                        width: isSelected ? 1.5 : 1,
                      ),
                    ),
                    child: Text(
                      s.toUpperCase(),
                      style: TextStyle(
                        color: isSelected ? color : Colors.grey,
                        fontWeight: FontWeight.w600,
                        fontSize: 13,
                      ),
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
          const SizedBox(height: 8),

          // Scenario description
          if (_scenarioDescriptions.containsKey(scenario))
            Text(
              _scenarioDescriptions[scenario]!,
              style: TextStyle(
                color: _scenarioColors[scenario]!.withOpacity(0.8),
                fontSize: 12,
              ),
            ),
          const SizedBox(height: 20),

          // Forecast chart
          SizedBox(
            height: 380,
            child: historicalAsync.when(
              data: (historical) => forecastAsync.when(
                data: (forecast) => ForecastChart(
                  historical: historical,
                  forecast: forecast,
                  scenario: scenario,
                ),
                loading: () => const _LoadingCard(height: 380),
                error: (e, _) => _ErrorCard(message: e.toString()),
              ),
              loading: () => const _LoadingCard(height: 380),
              error: (e, _) => _ErrorCard(message: e.toString()),
            ),
          ),
          const SizedBox(height: 24),

          // Scenario assumptions table
          _ScenarioAssumptions(scenario: scenario),
        ],
      ),
    );
  }
}

class _ScenarioAssumptions extends StatelessWidget {
  const _ScenarioAssumptions({required this.scenario});

  final String scenario;

  static const _assumptions = {
    'bear': [
      ('GHS/USD', '20.0'),
      ('Inflation', '31%'),
      ('Gold', '\$1,900'),
      ('Cocoa', '\$4,000/MT'),
    ],
    'base': [
      ('GHS/USD', '15.0'),
      ('Inflation', '20%'),
      ('Gold', '\$2,250'),
      ('Cocoa', '\$5,500/MT'),
    ],
    'bull': [
      ('GHS/USD', '11.0'),
      ('Inflation', '12%'),
      ('Gold', '\$2,600'),
      ('Cocoa', '\$7,000/MT'),
    ],
  };

  @override
  Widget build(BuildContext context) {
    final rows = _assumptions[scenario] ?? [];
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF30363D)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '${scenario.toUpperCase()} scenario assumptions (2025–2026)',
            style: const TextStyle(
                color: Colors.white70,
                fontSize: 13,
                fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 12),
          ...rows.map((r) => Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                  children: [
                    SizedBox(
                        width: 120,
                        child: Text(r.$1,
                            style: const TextStyle(
                                color: Colors.grey, fontSize: 13))),
                    Text(r.$2,
                        style: const TextStyle(
                            color: Colors.white,
                            fontSize: 13,
                            fontWeight: FontWeight.w500)),
                  ],
                ),
              )),
        ],
      ),
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
