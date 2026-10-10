package com.argus.dto;

import com.argus.entity.ChatMessage;
import com.argus.entity.MessageSender;

import java.time.Instant;
import java.util.UUID;

public record ChatMessageResponse(
        UUID id,
        MessageSender sender,
        String content,
        String resultPayload,
        Instant createdAt
) {
    public static ChatMessageResponse from(ChatMessage m) {
        return new ChatMessageResponse(m.getId(), m.getSender(), m.getContent(), m.getResultPayload(), m.getCreatedAt());
    }
}
