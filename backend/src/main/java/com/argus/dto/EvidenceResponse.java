package com.argus.dto;

import com.argus.entity.Evidence;
import com.argus.entity.EvidenceType;
import com.argus.entity.ProcessingStatus;

import java.time.Instant;
import java.util.UUID;

public record EvidenceResponse(
        UUID id,
        EvidenceType type,
        String originalFilename,
        String sourceLabel,
        String sha256Hash,
        Long sizeBytes,
        String mimeType,
        ProcessingStatus processingStatus,
        Integer processingProgress,
        Instant createdAt
) {
    public static EvidenceResponse from(Evidence e) {
        return new EvidenceResponse(
                e.getId(), e.getType(), e.getOriginalFilename(), e.getSourceLabel(),
                e.getSha256Hash(), e.getSizeBytes(), e.getMimeType(),
                e.getProcessingStatus(), e.getProcessingProgress(), e.getCreatedAt()
        );
    }
}
