package com.argus.dto;

import com.argus.entity.Investigation;
import com.argus.entity.InvestigationStatus;

import java.time.Instant;
import java.time.LocalDate;
import java.util.UUID;

public record InvestigationResponse(
        UUID id,
        String name,
        String description,
        LocalDate incidentDate,
        String location,
        InvestigationStatus status,
        Instant createdAt,
        Instant updatedAt,
        long evidenceCount,
        long findingsCount
) {
    public static InvestigationResponse from(Investigation inv, long evidenceCount, long findingsCount) {
        return new InvestigationResponse(
                inv.getId(),
                inv.getName(),
                inv.getDescription(),
                inv.getIncidentDate(),
                inv.getLocation(),
                inv.getStatus(),
                inv.getCreatedAt(),
                inv.getUpdatedAt(),
                evidenceCount,
                findingsCount
        );
    }
}
