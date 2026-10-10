package com.argus.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.util.Map;
import java.util.UUID;

/**
 * Thin client to the Python AI service (see /ai-service).
 *
 * Phase 1: the AI service only exposes a stub /query endpoint that returns a
 * canned response so the full request path (React -> Spring -> Python -> back)
 * exists end-to-end. Phase 2/3 will replace the stub with real video
 * processing, embeddings and hybrid retrieval — this client's interface is
 * intentionally kept generic so callers don't need to change later.
 */
@Component
public class AiServiceClient {

    private final RestClient restClient;

    public AiServiceClient(@Value("${argus.ai-service.base-url}") String baseUrl) {
        this.restClient = RestClient.builder().baseUrl(baseUrl).build();
    }

    public AiQueryResult query(UUID investigationId, String message) {
        try {
            return restClient.post()
                    .uri("/query")
                    .body(Map.of(
                            "investigation_id", investigationId.toString(),
                            "message", message
                    ))
                    .retrieve()
                    .body(AiQueryResult.class);
        } catch (Exception e) {
            // The AI service being unreachable should never break the chat -
            // degrade gracefully instead of a 500.
            return new AiQueryResult(
                    "Argus's analysis service is unavailable right now. Your message has been saved; try again shortly.",
                    null
            );
        }
    }

    public record AiQueryResult(String reply, Object results) {}
}
