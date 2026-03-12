/// Data models for the AHPI Flutter application.
library;

// ---------------------------------------------------------------------------
// AHPI Index point
// ---------------------------------------------------------------------------

class AhpiPoint {
  const AhpiPoint({required this.date, required this.value});

  final String date;
  final double value;

  factory AhpiPoint.fromJson(Map<String, dynamic> json) => AhpiPoint(
        date: json['ds'] as String,
        value: (json['y'] as num).toDouble(),
      );
}

// ---------------------------------------------------------------------------
// Forecast point (yhat + confidence interval)
// ---------------------------------------------------------------------------

class ForecastPoint {
  const ForecastPoint({
    required this.date,
    required this.yhat,
    required this.yhatLower,
    required this.yhatUpper,
  });

  final String date;
  final double yhat;
  final double yhatLower;
  final double yhatUpper;

  factory ForecastPoint.fromJson(Map<String, dynamic> json) => ForecastPoint(
        date: json['ds'] as String,
        yhat: (json['yhat'] as num).toDouble(),
        yhatLower: (json['yhat_lower'] as num?)?.toDouble() ?? (json['yhat'] as num).toDouble(),
        yhatUpper: (json['yhat_upper'] as num?)?.toDouble() ?? (json['yhat'] as num).toDouble(),
      );
}

// ---------------------------------------------------------------------------
// District / Prime index point
// ---------------------------------------------------------------------------

class LocationPoint {
  const LocationPoint({
    required this.date,
    required this.value,
    required this.location,
  });

  final String date;
  final double value;
  final String location;

  factory LocationPoint.fromJson(Map<String, dynamic> json, {String locationKey = 'district'}) =>
      LocationPoint(
        date: json['ds'] as String,
        value: (json['y'] as num).toDouble(),
        location: json[locationKey] as String,
      );
}

// ---------------------------------------------------------------------------
// Summary stats
// ---------------------------------------------------------------------------

class AhpiSummary {
  const AhpiSummary({
    required this.periodStart,
    required this.periodEnd,
    required this.ahpiStart,
    required this.ahpiEnd,
    required this.ahpiPctChange,
    required this.fxStart,
    required this.fxEnd,
    required this.fxPctChange,
    required this.usdPriceStart,
    required this.usdPriceEnd,
    required this.usdPricePctChange,
    required this.observations,
  });

  final String periodStart;
  final String periodEnd;
  final double ahpiStart;
  final double ahpiEnd;
  final double ahpiPctChange;
  final double fxStart;
  final double fxEnd;
  final double fxPctChange;
  final double usdPriceStart;
  final double usdPriceEnd;
  final double usdPricePctChange;
  final int observations;

  factory AhpiSummary.fromJson(Map<String, dynamic> json) {
    final period = json['period'] as Map<String, dynamic>;
    final ahpi = json['ahpi'] as Map<String, dynamic>;
    final fx = json['exchange_rate'] as Map<String, dynamic>;
    final usd = json['price_usd_per_sqm'] as Map<String, dynamic>;
    return AhpiSummary(
      periodStart: period['start'] as String,
      periodEnd: period['end'] as String,
      ahpiStart: (ahpi['start'] as num).toDouble(),
      ahpiEnd: (ahpi['end'] as num).toDouble(),
      ahpiPctChange: (ahpi['pct_change'] as num).toDouble(),
      fxStart: (fx['start'] as num).toDouble(),
      fxEnd: (fx['end'] as num).toDouble(),
      fxPctChange: (fx['pct_change'] as num).toDouble(),
      usdPriceStart: (usd['start'] as num).toDouble(),
      usdPriceEnd: (usd['end'] as num).toDouble(),
      usdPricePctChange: (usd['pct_change'] as num).toDouble(),
      observations: json['observations'] as int,
    );
  }
}

// ---------------------------------------------------------------------------
// Location summary (district or prime area)
// ---------------------------------------------------------------------------

class LocationSummary {
  const LocationSummary({
    required this.name,
    required this.latestAhpi,
    required this.startAhpi,
    required this.pctChange,
    required this.latestDate,
  });

  final String name;
  final double latestAhpi;
  final double startAhpi;
  final double pctChange;
  final String latestDate;

  factory LocationSummary.fromJson(Map<String, dynamic> json, {String nameKey = 'district'}) =>
      LocationSummary(
        name: json[nameKey] as String,
        latestAhpi: (json['latest_ahpi'] as num).toDouble(),
        startAhpi: (json['start_ahpi'] as num).toDouble(),
        pctChange: (json['pct_change'] as num).toDouble(),
        latestDate: json['latest_date'] as String,
      );
}

// ---------------------------------------------------------------------------
// Auth user
// ---------------------------------------------------------------------------

class AuthUser {
  const AuthUser({
    required this.sub,
    required this.email,
    required this.name,
    required this.pictureUrl,
    required this.accessToken,
  });

  final String sub;
  final String email;
  final String name;
  final String pictureUrl;
  final String accessToken;
}

// ---------------------------------------------------------------------------
// Async state wrapper
// ---------------------------------------------------------------------------

enum AsyncStatus { initial, loading, success, failure }

class AsyncState<T> {
  const AsyncState({
    this.status = AsyncStatus.initial,
    this.data,
    this.error,
  });

  final AsyncStatus status;
  final T? data;
  final String? error;

  bool get isLoading => status == AsyncStatus.loading;
  bool get hasData => status == AsyncStatus.success && data != null;
  bool get hasError => status == AsyncStatus.failure;

  AsyncState<T> copyWith({
    AsyncStatus? status,
    T? data,
    String? error,
  }) =>
      AsyncState<T>(
        status: status ?? this.status,
        data: data ?? this.data,
        error: error ?? this.error,
      );
}
