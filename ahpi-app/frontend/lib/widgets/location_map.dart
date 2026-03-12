import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import '../config/constants.dart';
import '../models/models.dart';

/// OpenStreetMap choropleth showing the latest AHPI value per location.
class LocationMap extends StatelessWidget {
  const LocationMap({
    super.key,
    required this.districtSummaries,
    required this.primeSummaries,
  });

  final List<LocationSummary> districtSummaries;
  final List<LocationSummary> primeSummaries;

  @override
  Widget build(BuildContext context) {
    final allSummaries = [...districtSummaries, ...primeSummaries];
    if (allSummaries.isEmpty) {
      return const Center(child: Text('No map data', style: TextStyle(color: Colors.grey)));
    }

    final maxAhpi =
        allSummaries.map((s) => s.latestAhpi).reduce((a, b) => a > b ? a : b);
    final minAhpi =
        allSummaries.map((s) => s.latestAhpi).reduce((a, b) => a < b ? a : b);

    Color colorForAhpi(double ahpi) {
      final t = (ahpi - minAhpi) / (maxAhpi - minAhpi).clamp(1.0, double.infinity);
      return Color.lerp(const Color(0xFF1565C0), const Color(0xFFFF6F00), t)!;
    }

    List<CircleMarker> buildMarkers(
      List<LocationSummary> summaries,
      Map<String, List<double>> coordMap,
    ) {
      return summaries.map((s) {
        final coords = coordMap[s.name];
        if (coords == null) return null;
        return CircleMarker(
          point: LatLng(coords[0], coords[1]),
          radius: 14,
          color: colorForAhpi(s.latestAhpi).withOpacity(0.8),
          borderColor: Colors.white,
          borderStrokeWidth: 1.5,
        );
      }).whereType<CircleMarker>().toList();
    }

    final markers = [
      ...buildMarkers(districtSummaries, AppConstants.districtCoordinates),
      ...buildMarkers(primeSummaries, AppConstants.primeCoordinates),
    ];

    List<Marker> buildLabelMarkers(
      List<LocationSummary> summaries,
      Map<String, List<double>> coordMap,
    ) {
      return summaries.map((s) {
        final coords = coordMap[s.name];
        if (coords == null) return null;
        return Marker(
          point: LatLng(coords[0], coords[1]),
          width: 120,
          height: 60,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const SizedBox(height: 28), // offset below circle
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                decoration: BoxDecoration(
                  color: const Color(0xCC0D1117),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  '${s.name}\n${s.latestAhpi.toStringAsFixed(0)}',
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 9,
                      fontWeight: FontWeight.w600),
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
        );
      }).whereType<Marker>().toList();
    }

    final labelMarkers = [
      ...buildLabelMarkers(districtSummaries, AppConstants.districtCoordinates),
      ...buildLabelMarkers(primeSummaries, AppConstants.primeCoordinates),
    ];

    return FlutterMap(
      options: MapOptions(
        initialCenter: LatLng(
          AppConstants.accraCenter[0],
          AppConstants.accraCenter[1],
        ),
        initialZoom: 11,
      ),
      children: [
        TileLayer(
          urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
          userAgentPackageName: 'com.ahpi.flutter',
        ),
        CircleLayer(circles: markers),
        MarkerLayer(markers: labelMarkers),
      ],
    );
  }
}
