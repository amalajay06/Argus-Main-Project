package com.argus.repository;

import com.argus.entity.Investigation;
import com.argus.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.UUID;

public interface InvestigationRepository extends JpaRepository<Investigation, UUID> {
    List<Investigation> findByCreatedByOrderByUpdatedAtDesc(User createdBy);
}
