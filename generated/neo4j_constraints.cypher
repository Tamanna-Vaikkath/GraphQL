// GENERATED from ontology/gl_claims.yaml v0.2.0 — do not edit by hand

CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (n:Policy) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT broker_id IF NOT EXISTS FOR (n:Broker) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT insuredlocation_id IF NOT EXISTS FOR (n:InsuredLocation) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT coveragepart_id IF NOT EXISTS FOR (n:CoveragePart) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (n:Claim) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT occurrence_id IF NOT EXISTS FOR (n:Occurrence) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT litigation_id IF NOT EXISTS FOR (n:Litigation) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT feature_id IF NOT EXISTS FOR (n:Feature) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT injury_id IF NOT EXISTS FOR (n:Injury) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT reserve_id IF NOT EXISTS FOR (n:Reserve) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT payment_id IF NOT EXISTS FOR (n:Payment) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT recovery_id IF NOT EXISTS FOR (n:Recovery) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT party_id IF NOT EXISTS FOR (n:Party) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT insured_id IF NOT EXISTS FOR (n:Insured) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT adjuster_id IF NOT EXISTS FOR (n:Adjuster) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT tpa_id IF NOT EXISTS FOR (n:TPA) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT claimant_id IF NOT EXISTS FOR (n:Claimant) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT attorney_id IF NOT EXISTS FOR (n:Attorney) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT medicalprovider_id IF NOT EXISTS FOR (n:MedicalProvider) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT extractedfact_id IF NOT EXISTS FOR (n:ExtractedFact) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ingestionrun_id IF NOT EXISTS FOR (n:IngestionRun) REQUIRE n.id IS UNIQUE;

// Filtered vector index on chunks (used by the vector step; validate syntax against your Neo4j / Aura version)
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (ch:Chunk) ON ch.embedding
OPTIONS { indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: 'cosine' } };