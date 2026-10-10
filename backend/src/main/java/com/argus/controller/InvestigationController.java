package com.argus.controller;

import com.argus.dto.InvestigationRequest;
import com.argus.dto.InvestigationResponse;
import com.argus.service.InvestigationService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/investigations")
public class InvestigationController {

    private final InvestigationService investigationService;

    public InvestigationController(InvestigationService investigationService) {
        this.investigationService = investigationService;
    }

    @GetMapping
    public List<InvestigationResponse> list() {
        return investigationService.listForCurrentUser();
    }

    @PostMapping
    public InvestigationResponse create(@Valid @RequestBody InvestigationRequest request) {
        return investigationService.create(request);
    }

    @GetMapping("/{id}")
    public InvestigationResponse get(@PathVariable UUID id) {
        return investigationService.getResponse(id);
    }
}
