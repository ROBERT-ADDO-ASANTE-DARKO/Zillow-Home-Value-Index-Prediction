import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:intl/intl.dart';
import 'package:latlong2/latlong.dart';
import 'package:provider/provider.dart';

import '../providers/app_provider.dart';

class MapSection extends StatelessWidget {
  const MapSection({super.key});

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<AppProvider>();
    final coords = provider.cityCoordinates;

    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Section header ───────────────────────────────────────────────
            Row(
              children: [
                const Icon(Icons.map, color: Color(0xFF1565C0), size: 22),
                const SizedBox(width: 10),
                Text(
                  'Geographic View'
                  '${provider.selectedCity != null ? ' – ${provider.selectedCity}' : ''}',
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1A237E),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),

            // ── Map ──────────────────────────────────────────────────────────
            if (provider.dataState == LoadingState.loading)
              const SizedBox(
                height: 350,
                child: Center(child: CircularProgressIndicator()),
              )
            else if (coords == null)
              const SizedBox(
                height: 200,
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.location_off, color: Colors.grey, size: 48),
                      SizedBox(height: 8),
                      Text('Coordinates not available for this city',
                          style: TextStyle(color: Colors.grey)),
                    ],
                  ),
                ),
              )
            else
              ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: SizedBox(
                  height: 380,
                  child: FlutterMap(
                    options: MapOptions(
                      initialCenter:
                          LatLng(coords['lat']!, coords['lon']!),
                      initialZoom: 11,
                    ),
                    children: [
                      TileLayer(
                        urlTemplate:
                            'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                        userAgentPackageName: 'com.zhvi.prediction',
                      ),
                      MarkerLayer(
                        markers: _buildMarkers(provider),
                      ),
                    ],
                  ),
                ),
              ),

            // ── Zipcode legend ───────────────────────────────────────────────
            if (coords != null && provider.cityMarkers.isNotEmpty) ...[
              const SizedBox(height: 12),
              Text(
                '${provider.cityMarkers.length} zipcode(s) in ${provider.selectedCity ?? ''}',
                style: TextStyle(color: Colors.grey[600], fontSize: 13),
              ),
            ],
          ],
        ),
      ),
    );
  }

  List<Marker> _buildMarkers(AppProvider provider) {
    if (provider.cityMarkers.isEmpty) return [];

    // Deduplicate: show one representative marker per unique coordinate
    // (all zipcodes in a city share the same city coordinates)
    final seen = <String>{};
    final markers = <Marker>[];

    for (final m in provider.cityMarkers) {
      final key = '${m.lat}_${m.lon}';
      if (!seen.add(key)) continue;

      markers.add(
        Marker(
          point: LatLng(m.lat, m.lon),
          width: 180,
          height: 60,
          child: _CityMarkerWidget(
            city: provider.selectedCity ?? '',
            latestValue: m.latestValue,
          ),
        ),
      );
    }
    return markers;
  }
}

class _CityMarkerWidget extends StatelessWidget {
  final String city;
  final double latestValue;

  const _CityMarkerWidget({required this.city, required this.latestValue});

  @override
  Widget build(BuildContext context) {
    final fmt = NumberFormat.currency(symbol: '\$', decimalDigits: 0);
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration(
            color: const Color(0xFF1565C0),
            borderRadius: BorderRadius.circular(6),
          ),
          child: Text(
            '$city\n${fmt.format(latestValue)}',
            textAlign: TextAlign.center,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 11,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        const Icon(Icons.location_pin, color: Color(0xFF1565C0), size: 24),
      ],
    );
  }
}
