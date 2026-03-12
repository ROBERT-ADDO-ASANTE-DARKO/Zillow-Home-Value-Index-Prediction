import 'package:flutter/material.dart';

import '../models/models.dart';

/// Displays 4 top-line KPI cards derived from [AhpiSummary].
class MarketKpis extends StatelessWidget {
  const MarketKpis({super.key, required this.summary});

  final AhpiSummary summary;

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 12,
      runSpacing: 12,
      children: [
        _KpiCard(
          label: 'AHPI (Dec 2024)',
          value: summary.ahpiEnd.toStringAsFixed(1),
          subtitle: 'Base 2015 = 100',
          change: summary.ahpiPctChange,
          icon: Icons.trending_up,
          color: const Color(0xFF4CAF50),
        ),
        _KpiCard(
          label: 'GHS/USD Rate',
          value: summary.fxEnd.toStringAsFixed(2),
          subtitle: 'Exchange rate',
          change: summary.fxPctChange,
          icon: Icons.currency_exchange,
          color: const Color(0xFFFFD54F),
        ),
        _KpiCard(
          label: 'Price (USD/sqm)',
          value: '\$${summary.usdPriceEnd.toStringAsFixed(0)}',
          subtitle: 'Mid-market avg',
          change: summary.usdPricePctChange,
          icon: Icons.home_outlined,
          color: const Color(0xFF64B5F6),
        ),
        _KpiCard(
          label: 'Data Points',
          value: summary.observations.toString(),
          subtitle: '${summary.periodStart} – ${summary.periodEnd}',
          change: null,
          icon: Icons.bar_chart,
          color: const Color(0xFFCE93D8),
        ),
      ],
    );
  }
}

class _KpiCard extends StatelessWidget {
  const _KpiCard({
    required this.label,
    required this.value,
    required this.subtitle,
    required this.change,
    required this.icon,
    required this.color,
  });

  final String label;
  final String value;
  final String subtitle;
  final double? change;
  final IconData icon;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final isPositive = (change ?? 0) >= 0;

    return Container(
      width: 180,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF30363D)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Icon(icon, color: color, size: 20),
              if (change != null)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: isPositive
                        ? Colors.green.withOpacity(0.15)
                        : Colors.red.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    '${isPositive ? '+' : ''}${change!.toStringAsFixed(1)}%',
                    style: TextStyle(
                      color: isPositive ? Colors.greenAccent : Colors.redAccent,
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 22,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: const TextStyle(color: Colors.white70, fontSize: 12),
          ),
          Text(
            subtitle,
            style: TextStyle(color: Colors.grey[600], fontSize: 11),
          ),
        ],
      ),
    );
  }
}
