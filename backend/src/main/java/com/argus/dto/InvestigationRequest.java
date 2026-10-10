package com.argus.dto;

import jakarta.validation.constraints.NotBlank;

import java.time.LocalDate;

public record InvestigationRequest(
        @NotBlank String name,
        String description,
        LocalDate incidentDate,
        String location
) {}
