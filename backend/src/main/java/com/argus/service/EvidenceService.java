package com.argus.service;

import com.argus.dto.EvidenceResponse;
import com.argus.entity.Evidence;
import com.argus.entity.EvidenceType;
import com.argus.entity.Investigation;
import com.argus.entity.ProcessingStatus;
import com.argus.repository.EvidenceRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;
import java.util.UUID;

@Service
public class EvidenceService {

    private final EvidenceRepository evidenceRepository;
    private final InvestigationService investigationService;
    private final CurrentUserService currentUserService;
    private final Path evidenceRoot;

    public EvidenceService(
            EvidenceRepository evidenceRepository,
            InvestigationService investigationService,
            CurrentUserService currentUserService,
            @Value("${argus.storage.evidence-root}") String evidenceRoot
    ) {
        this.evidenceRepository = evidenceRepository;
        this.investigationService = investigationService;
        this.currentUserService = currentUserService;
        this.evidenceRoot = Path.of(evidenceRoot);
    }

    public EvidenceResponse upload(UUID investigationId, MultipartFile file, String sourceLabel) {
        Investigation investigation = investigationService.getOwned(investigationId);

        if (file.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Uploaded file is empty");
        }

        try {
            Path caseDir = evidenceRoot.resolve(investigationId.toString());
            Files.createDirectories(caseDir);

            String storedName = UUID.randomUUID() + "_" + sanitizeFilename(file.getOriginalFilename());
            Path destination = caseDir.resolve(storedName);

            String sha256;
            try (InputStream in = file.getInputStream()) {
                sha256 = hashAndCopy(in, destination);
            }

            Evidence evidence = Evidence.builder()
                    .investigation(investigation)
                    .type(classify(file.getContentType()))
                    .originalFilename(file.getOriginalFilename())
                    .storagePath(destination.toString())
                    .sha256Hash(sha256)
                    .sizeBytes(file.getSize())
                    .mimeType(file.getContentType())
                    .sourceLabel(sourceLabel)
                    .processingStatus(ProcessingStatus.PENDING)
                    .processingProgress(0)
                    .uploadedBy(currentUserService.get())
                    .build();

            evidenceRepository.save(evidence);

            // Phase 2 will enqueue this evidence for the Python AI pipeline
            // (detection/tracking/embeddings). For now it stays PENDING.

            return EvidenceResponse.from(evidence);
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to store evidence file", e);
        }
    }

    public List<EvidenceResponse> list(UUID investigationId) {
        Investigation investigation = investigationService.getOwned(investigationId);
        return evidenceRepository.findByInvestigationOrderByCreatedAtDesc(investigation).stream()
                .map(EvidenceResponse::from)
                .toList();
    }

    private String hashAndCopy(InputStream in, Path destination) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (DigestInputStream dis = new DigestInputStream(in, digest)) {
                Files.copy(dis, destination, StandardCopyOption.REPLACE_EXISTING);
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 not available on this JVM", e);
        }
    }

    private EvidenceType classify(String mimeType) {
        if (mimeType == null) return EvidenceType.DOCUMENT;
        if (mimeType.startsWith("video/")) return EvidenceType.VIDEO;
        if (mimeType.startsWith("image/")) return EvidenceType.IMAGE;
        return EvidenceType.DOCUMENT;
    }

    private String sanitizeFilename(String filename) {
        if (filename == null) return "evidence";
        return filename.replaceAll("[^a-zA-Z0-9._-]", "_");
    }
}
