package com.argus.dto;

public record AuthResponse(
        String token,
        String username,
        String fullName
) {}
