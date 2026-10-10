package com.argus.repository;

import com.argus.entity.Evidence;
import com.argus.entity.Investigation;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.UUID;

public interface EvidenceRepository extends JpaRepository<Evidence, UUID> {
    List<Evidence> findByInvestigationOrderByCreatedAtDesc(Investigation investigation);
}
