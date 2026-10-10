package com.argus.repository;

import com.argus.entity.ChatMessage;
import com.argus.entity.Investigation;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.UUID;

public interface ChatMessageRepository extends JpaRepository<ChatMessage, UUID> {
    List<ChatMessage> findByInvestigationOrderByCreatedAtAsc(Investigation investigation);
}
