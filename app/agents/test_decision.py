from app.agents.decision import make_decision


RESEARCH_QUESTION = (
    "How can AI improve customer support efficiency?"
)


# =============================================================
# TEST DATA
# =============================================================

def create_research_analysis():

    return {

        "research_question":
            RESEARCH_QUESTION,

        "key_findings": [

            "AI can automate routine "
            "customer support tasks.",

            "AI chatbots can assist with "
            "low-complexity queries."
        ],

        "supporting_evidence": [

            "AI agents can automate routine "
            "support tasks."
        ],

        "limitations": [

            "Human agents are still needed "
            "for complex cases."
        ],

        "confidence":
            "Medium",
    }


def create_supporting_verification():

    return {

        "research_question":
            RESEARCH_QUESTION,

        "overall_relevance":
            "High",

        "verification_confidence":
            "High",

        "limitations":
            [],

        "evidence_assessment": [

            {

                "evidence_id":
                    "E001",

                "relationship":
                    "Supports",

                "claim_supported":
                    (
                        "AI can automate routine "
                        "support tasks."
                    ),

                "reasoning":
                    (
                        "The evidence directly "
                        "supports the finding."
                    ),
            }
        ],
    }


# =============================================================
# TEST 1
# SUPPORTING EVIDENCE
# =============================================================

def test_make_decision_with_supporting_evidence():

    result = make_decision(

        research_question=
            RESEARCH_QUESTION,

        research_analysis=
            create_research_analysis(),

        verification=
            create_supporting_verification(),
    )

    assert (
        result["research_question"]
        ==
        RESEARCH_QUESTION
    )

    assert (
        result["decision_status"]
        ==
        "supported_evidence"
    )

    assert (
        "AI"
        in
        result["decision"]
    )

    assert (
        RESEARCH_QUESTION
        in
        result["decision"]
    )

    assert (
        len(result["reasons"])
        >
        0
    )

    assert (
        result["confidence"]
        ==
        "High"
    )


# =============================================================
# TEST 2
# PARTIALLY SUPPORTING EVIDENCE
# =============================================================

def test_make_decision_with_partially_supporting_evidence():

    verification = (
        create_supporting_verification()
    )

    verification[
        "evidence_assessment"
    ][0][
        "relationship"
    ] = (
        "Partially Supports"
    )

    result = make_decision(

        research_question=
            RESEARCH_QUESTION,

        research_analysis=
            create_research_analysis(),

        verification=
            verification,
    )

    assert (
        result["decision_status"]
        ==
        "partially_supported_evidence"
    )

    assert (
        RESEARCH_QUESTION
        in
        result["decision"]
    )


# =============================================================
# TEST 3
# INSUFFICIENT EVIDENCE
# =============================================================

def test_make_decision_when_evidence_is_insufficient():

    verification = {

        "research_question":
            RESEARCH_QUESTION,

        "overall_relevance":
            "Low",

        "verification_confidence":
            "Low",

        "limitations":
            [],

        "evidence_assessment": [

            {

                "evidence_id":
                    "E001",

                "relationship":
                    "Neutral",
            }
        ],
    }

    result = make_decision(

        research_question=
            RESEARCH_QUESTION,

        research_analysis=
            create_research_analysis(),

        verification=
            verification,
    )

    assert (
        result["decision_status"]
        ==
        "insufficient_evidence"
    )

    assert (
        "insufficient"
        in
        result["decision"].lower()
    )


# =============================================================
# TEST 4
# CONTRADICTING EVIDENCE
# =============================================================

def test_make_decision_with_contradicting_evidence():

    verification = {

        "research_question":
            RESEARCH_QUESTION,

        "overall_relevance":
            "High",

        "verification_confidence":
            "Medium",

        "limitations":
            [],

        "evidence_assessment": [

            {

                "evidence_id":
                    "E001",

                "relationship":
                    "Contradicts",
            }
        ],
    }

    result = make_decision(

        research_question=
            RESEARCH_QUESTION,

        research_analysis=
            create_research_analysis(),

        verification=
            verification,
    )

    assert (
        result["decision_status"]
        ==
        "insufficient_evidence"
    )


# =============================================================
# TEST 5
# CONFLICTING EVIDENCE
# =============================================================

def test_decision_handles_conflicting_evidence():

    verification = {

        "research_question":
            RESEARCH_QUESTION,

        "overall_relevance":
            "High",

        "verification_confidence":
            "High",

        "limitations":
            [],

        "evidence_assessment": [

            {

                "evidence_id":
                    "E001",

                "relationship":
                    "Supports",
            },

            {

                "evidence_id":
                    "E002",

                "relationship":
                    "Contradicts",
            }
        ],
    }

    result = make_decision(

        research_question=
            RESEARCH_QUESTION,

        research_analysis=
            create_research_analysis(),

        verification=
            verification,
    )

    assert (
        result["decision_status"]
        ==
        "conflicted_evidence"
    )

    assert (
        "conflicting"
        in
        result["decision"].lower()
    )


# =============================================================
# TEST 6
# RISKS AND ALTERNATIVES
# =============================================================

def test_decision_contains_risks_and_alternatives():

    result = make_decision(

        research_question=
            RESEARCH_QUESTION,

        research_analysis=
            create_research_analysis(),

        verification=
            create_supporting_verification(),
    )

    assert (
        len(result["risks"])
        >
        0
    )

    assert (
        len(result["alternatives"])
        >
        0
    )


# =============================================================
# TEST 7
# NOT HARDCODED
# =============================================================

def test_decision_is_not_hardcoded_to_customer_support():

    question = (
        "What are the benefits of renewable energy?"
    )

    analysis = {

        "key_findings": [

            "Renewable energy can reduce "
            "dependence on fossil fuels."
        ],

        "limitations": [

            "Energy production can depend "
            "on weather conditions."
        ]
    }

    verification = {

        "overall_relevance":
            "High",

        "verification_confidence":
            "High",

        "evidence_assessment": [

            {

                "evidence_id":
                    "E001",

                "relationship":
                    "Supports"
            }
        ]
    }

    result = make_decision(

        research_question=
            question,

        research_analysis=
            analysis,

        verification=
            verification,
    )

    assert (
        question
        in
        result["decision"]
    )

    assert (
        "customer support"
        not in
        result["decision"].lower()
    )