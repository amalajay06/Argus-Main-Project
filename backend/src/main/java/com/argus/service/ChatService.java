package com.argus.service;

import com.argus.dto.ChatMessageResponse;
import com.argus.entity.ChatMessage;
import com.argus.entity.Investigation;
import com.argus.entity.MessageSender;
import com.argus.repository.ChatMessageRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.UUID;

@Service
public class ChatService {

    private final ChatMessageRepository chatMessageRepository;
    private final InvestigationService investigationService;
    private final AiServiceClient aiServiceClient;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public ChatService(
            ChatMessageRepository chatMessageRepository,
            InvestigationService investigationService,
            AiServiceClient aiServiceClient
    ) {
        this.chatMessageRepository = chatMessageRepository;
        this.investigationService = investigationService;
        this.aiServiceClient = aiServiceClient;
    }

    public List<ChatMessageResponse> history(UUID investigationId) {
        Investigation investigation = investigationService.getOwned(investigationId);
        return chatMessageRepository.findByInvestigationOrderByCreatedAtAsc(investigation).stream()
                .map(ChatMessageResponse::from)
                .toList();
    }

    public ChatMessageResponse sendMessage(UUID investigationId, String content) {
        Investigation investigation = investigationService.getOwned(investigationId);

        ChatMessage userMessage = ChatMessage.builder()
                .investigation(investigation)
                .sender(MessageSender.USER)
                .content(content)
                .build();
        chatMessageRepository.save(userMessage);

        AiServiceClient.AiQueryResult aiResult = aiServiceClient.query(investigationId, content);

        String resultPayloadJson = null;
        if (aiResult.results() != null) {
            try {
                resultPayloadJson = objectMapper.writeValueAsString(aiResult.results());
            } catch (JsonProcessingException ignored) {
                // If serialization fails we still want the text reply to reach the investigator.
            }
        }

        ChatMessage argusMessage = ChatMessage.builder()
                .investigation(investigation)
                .sender(MessageSender.ARGUS)
                .content(aiResult.reply())
                .resultPayload(resultPayloadJson)
                .build();
        chatMessageRepository.save(argusMessage);

        return ChatMessageResponse.from(argusMessage);
    }
}
