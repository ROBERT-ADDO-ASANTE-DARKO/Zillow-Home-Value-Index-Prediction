/// Data models for the ZHVI Prediction application.

class MarketMetrics {
  final double volatility;
  final double roi;
  final double riskScore;

  const MarketMetrics({
    required this.volatility,
    required this.roi,
    required this.riskScore,
  });

  factory MarketMetrics.fromJson(Map<String, dynamic> json) => MarketMetrics(
        volatility: (json['volatility'] as num).toDouble(),
        roi: (json['roi'] as num).toDouble(),
        riskScore: (json['risk_score'] as num).toDouble(),
      );
}

class PricePoint {
  final DateTime date;
  final double value;

  const PricePoint({required this.date, required this.value});

  factory PricePoint.fromJson(Map<String, dynamic> json) => PricePoint(
        date: DateTime.parse(json['date'] as String),
        value: (json['value'] as num).toDouble(),
      );
}

class ForecastResult {
  final List<DateTime> dates;
  final List<double> forecast;
  final List<double> lower;
  final List<double> upper;

  const ForecastResult({
    required this.dates,
    required this.forecast,
    required this.lower,
    required this.upper,
  });

  factory ForecastResult.fromJson(Map<String, dynamic> json) => ForecastResult(
        dates: (json['dates'] as List<dynamic>)
            .map((d) => DateTime.parse(d as String))
            .toList(),
        forecast: (json['forecast'] as List<dynamic>)
            .map((v) => (v as num).toDouble())
            .toList(),
        lower: (json['lower'] as List<dynamic>)
            .map((v) => (v as num).toDouble())
            .toList(),
        upper: (json['upper'] as List<dynamic>)
            .map((v) => (v as num).toDouble())
            .toList(),
      );
}

class CityMarker {
  final String zipcode;
  final double lat;
  final double lon;
  final double latestValue;

  const CityMarker({
    required this.zipcode,
    required this.lat,
    required this.lon,
    required this.latestValue,
  });

  factory CityMarker.fromJson(Map<String, dynamic> json) => CityMarker(
        zipcode: json['zipcode'].toString(),
        lat: (json['lat'] as num).toDouble(),
        lon: (json['lon'] as num).toDouble(),
        latestValue: (json['latest_value'] as num).toDouble(),
      );
}
