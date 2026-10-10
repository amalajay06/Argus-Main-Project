package com.argus.controller;

import com.argus.dto.EvidenceResponse;
import com.argus.service.EvidenceService;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/investigations/{investigationId}/evidence")
public class EvidenceController {

    private final EvidenceService evidenceService;

    public EvidenceController(EvidenceService evidenceService) {
        this.evidenceService = evidenceService;
    }

    @GetMapping
    public List<EvidenceResponse> list(@PathVariable UUID investigationId) {
        return evidenceService.list(investigationId);
    }

    @PostMapping(consumes = "multipart/form-data")
    public EvidenceResponse upload(
            @PathVariable UUID investigationId,
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "sourceLabel", required = false) String sourceLabel
    ) {
        return evidenceService.upload(investigationId, file, sourceLabel);
    }
}
