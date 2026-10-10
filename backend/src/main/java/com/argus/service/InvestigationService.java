package com.argus.service;

import com.argus.dto.InvestigationRequest;
import com.argus.dto.InvestigationResponse;
import com.argus.entity.Investigation;
import com.argus.entity.User;
import com.argus.repository.ChatMessageRepository;
import com.argus.repository.EvidenceRepository;
import com.argus.repository.InvestigationRepository;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;
import java.util.UUID;

@Service
public class InvestigationService {

    private final InvestigationRepository investigationRepository;
    private final EvidenceRepository evidenceRepository;
    private final ChatMessageRepository chatMessageRepository;
    private final CurrentUserService currentUserService;

    public InvestigationService(
            InvestigationRepository investigationRepository,
            EvidenceRepository evidenceRepository,
            ChatMessageRepository chatMessageRepository,
            CurrentUserService currentUserService
    ) {
        this.investigationRepository = investigationRepository;
        this.evidenceRepository = evidenceRepository;
        this.chatMessageRepository = chatMessageRepository;
        this.currentUserService = currentUserService;
    }

    public InvestigationResponse create(InvestigationRequest request) {
        User user = currentUserService.get();

        Investigation investigation = Investigation.builder()
                .name(request.name())
                .description(request.description())
                .incidentDate(request.incidentDate())
                .location(request.location())
                .createdBy(user)
                .build();

        investigationRepository.save(investigation);
        return InvestigationResponse.from(investigation, 0, 0);
    }

    public List<InvestigationResponse> listForCurrentUser() {
        User user = currentUserService.get();
        return investigationRepository.findByCreatedByOrderByUpdatedAtDesc(user).stream()
                .map(inv -> InvestigationResponse.from(
                        inv,
                        evidenceRepository.findByInvestigationOrderByCreatedAtDesc(inv).size(),
                        0 // findings arrive in a later phase
                ))
                .toList();
    }

    public Investigation getOwned(UUID investigationId) {
        User user = currentUserService.get();
        Investigation investigation = investigationRepository.findById(investigationId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Investigation not found"));

        if (!investigation.getCreatedBy().getId().equals(user.getId())) {
            // Case-level data isolation (NFR in the PRD): investigators only see their own cases.
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Investigation not found");
        }
        return investigation;
    }

    public InvestigationResponse getResponse(UUID investigationId) {
        Investigation investigation = getOwned(investigationId);
        return InvestigationResponse.from(
                investigation,
                evidenceRepository.findByInvestigationOrderByCreatedAtDesc(investigation).size(),
                0
        );
    }
}
