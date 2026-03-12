import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/models.dart';
import '../services/api_service.dart';
import '../services/auth_service.dart';
import 'providers.dart';

// ---------------------------------------------------------------------------
// Services (singletons)
// ---------------------------------------------------------------------------

final authServiceProvider = Provider<AuthService>((ref) => AuthService());

final apiServiceProvider = Provider<ApiService>((ref) {
  final service = ApiService();
  // Keep access token in sync with the auth state
  ref.listen<AuthState>(authProvider, (_, next) {
    if (next.user != null) {
      service.setAccessToken(next.user!.accessToken);
    } else {
      service.clearAccessToken();
    }
  });
  return service;
});

// ---------------------------------------------------------------------------
// Auth state
// ---------------------------------------------------------------------------

class AuthState {
  const AuthState({
    this.user,
    this.isLoading = false,
    this.error,
  });

  final AuthUser? user;
  final bool isLoading;
  final String? error;

  bool get isAuthenticated => user != null;

  AuthState copyWith({
    AuthUser? user,
    bool? isLoading,
    String? error,
    bool clearUser = false,
    bool clearError = false,
  }) =>
      AuthState(
        user: clearUser ? null : (user ?? this.user),
        isLoading: isLoading ?? this.isLoading,
        error: clearError ? null : (error ?? this.error),
      );
}

// ---------------------------------------------------------------------------
// AuthNotifier
// ---------------------------------------------------------------------------

class AuthNotifier extends StateNotifier<AuthState> {
  AuthNotifier(this._authService) : super(const AuthState()) {
    _tryRestoreSession();
  }

  final AuthService _authService;

  Future<void> _tryRestoreSession() async {
    state = state.copyWith(isLoading: true, clearError: true);
    try {
      final user = await _authService.tryRestoreSession();
      state = state.copyWith(user: user, isLoading: false, clearUser: user == null);
    } catch (e) {
      state = state.copyWith(isLoading: false, clearUser: true);
    }
  }

  Future<void> login() async {
    state = state.copyWith(isLoading: true, clearError: true);
    try {
      final user = await _authService.login();
      state = state.copyWith(user: user, isLoading: false);
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Login failed: ${e.toString()}',
        clearUser: true,
      );
    }
  }

  Future<void> logout() async {
    state = state.copyWith(isLoading: true, clearError: true);
    try {
      await _authService.logout();
    } finally {
      state = const AuthState();
    }
  }
}

final authProvider = StateNotifierProvider<AuthNotifier, AuthState>(
  (ref) => AuthNotifier(ref.watch(authServiceProvider)),
);
