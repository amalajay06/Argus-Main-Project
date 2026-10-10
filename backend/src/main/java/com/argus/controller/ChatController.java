package com.argus.controller;

import com.argus.dto.ChatMessageRequest;
import com.argus.dto.ChatMessageResponse;
import com.argus.service.ChatService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/investigations/{investigationId}/messages")
public class ChatController {

    private final ChatService chatService;

    public ChatController(ChatService chatService) {
        this.chatService = chatService;
    }

    @GetMapping
    public List<ChatMessageResponse> history(@PathVariable UUID investigationId) {
        return chatService.history(investigationId);
    }

    @PostMapping
    public ChatMessageResponse send(
            @PathVariable UUID investigationId,
            @Valid @RequestBody ChatMessageRequest request
    ) {
        return chatService.sendMessage(investigationId, request.content());
    }
}
