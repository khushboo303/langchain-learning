from collections.abc import Set

import trajector.apps.drafts.chat.types as chat_types
from trajector.apps.drafts import configs
from trajector.apps.drafts import cores_api_types
from trajector.apps.drafts.chat import extractors
from trajector.apps.drafts.chat import text_snippets
from trajector.apps.drafts.chat import utils
from trajector.apps.drafts.chat import workflows
from trajector.apps.drafts.rest import types


class FunctionalMedicalImpairmentWorkflow(workflows.ChatWorkflow):
    """Set up pathing to load guidance templates for medical opinions"""

    _ALLOWED_MODELS: Set[configs.ProviderModel] = {
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0"}

    def __init__(self, config: configs.DraftsAppConfig, guidance_template_filename: str):
        template_path = config.guidance_path.joinpath(
            types.ScenarioDocumentType.FUNCTIONAL_IMPAIRMENT_EVALUATION.value,
            guidance_template_filename,
        )
        workflow_template = chat_types.GuidanceTemplate(
            template_path.read_text())
        super().__init__(config, workflow_template)
        self.add_data_dependency(
            "no_action_taken_msg", utils.NO_ACTION_TAKEN_MSG)
        self.add_data_dependency(
            "is_completion_actionable", utils.is_completion_actionable)


class CreateVeteranOntologyWorkflow(FunctionalMedicalImpairmentWorkflow):
    GUIDANCE_TEMPLATE = "scenario/create_veteran_ontology_workflow.guidance"

    def __init__(self, config: configs.DraftsAppConfig, fie_template: chat_types.MustacheTemplate):
        super().__init__(config, self.GUIDANCE_TEMPLATE)
        self.add_data_dependency("fie_template", fie_template)


async def generate_scenario_ontology(
    config, base_template: chat_types.MustacheTemplate
) -> chat_types.GeneratedFieScenarioOntologyParts:
    base_template, modifiers = extractors.extract_modifiers_from_template(
        base_template)

    vet_ont_workflow = CreateVeteranOntologyWorkflow(config, base_template)
    veteran_ontology = await vet_ont_workflow.execute()
    if utils.is_completion_actionable(veteran_ontology.value):
        ontology = veteran_ontology
    else:
        ontology = vet_ont_workflow.get_completion_by_name(
            "veteran_ontology_first_pass_json")

    return chat_types.GeneratedFieScenarioOntologyParts(
        content=types.CreateFieScenarioOntologyInContent(
            base_template=extractors.extract_intended_type_from_completion(
                base_template,
                types.ChatCompletionType.MUSTACHE,
                "template_statements",
            ),
            static_contents=modifiers[chat_types.DraftModifierSection.STATIC_CONTENT],
            veteran_ontology=extractors.extract_intended_type_from_completion(
                ontology.value,
                types.ChatCompletionType.JSON,
                veteran_ontology.instruction.name,
            ),
        ),
    )


class PopulateVeteranOntology(FunctionalMedicalImpairmentWorkflow):
    GUIDANCE_TEMPLATE = "draft/populate_veteran_ontology.guidance"

    def __init__(
        self,
        config: configs.DraftsAppConfig,
        veteran_ontology: chat_types.UnvalidatedJsonString,
        lay_statement_content: str,
        client_profile: cores_api_types.CoresClient,
    ):
        super().__init__(config, self.GUIDANCE_TEMPLATE)
        self.add_data_dependency("veteran_ontology", veteran_ontology)
        self.add_data_dependency(
            "lay_statement_content", lay_statement_content)
        self.add_data_dependency(
            "trajector_client_data", client_profile.model_dump(mode="json"))


class PopulateFieTemplateWorkflow(FunctionalMedicalImpairmentWorkflow):
    GUIDANCE_TEMPLATE = "draft/populate_fie_template_workflow.guidance"

    def __init__(
        self,
        config: configs.DraftsAppConfig,
        fie_template: str,
        client_profile: cores_api_types.CoresClient,
        populated_veteran_ontology: str,
    ):
        super().__init__(config, self.GUIDANCE_TEMPLATE)
        self.add_data_dependency("fie_template", fie_template)
        self.add_data_dependency(
            "trajector_client_data", client_profile.model_dump(mode="json"))
        self.add_data_dependency(
            "populated_veteran_ontology", populated_veteran_ontology)


async def generate_draft(
    config: configs.DraftsAppConfig,
    ontology_content: chat_types.FieScenarioOntologyContent,
    ls_draft: types.Draft,
    client_profile: cores_api_types.CoresClient,
) -> chat_types.GeneratedDraftParts:
    client_profile_snapshot_content = client_profile.model_dump(mode="json")
    # Remove SSN information before sending to LLM
    ssn = utils.extract_client_ssn(client_profile_snapshot_content)
    full_name = utils.extract_client_full_name(client_profile_snapshot_content)
    utils.remove_client_ssn(client_profile_snapshot_content)

    ls_content = ls_draft.content
    ls_content = text_snippets.mask_medical_text_snippet_if_exists(ls_content)

    populate_veteran_ontology = PopulateVeteranOntology(
        config, ontology_content.veteran_ontology, ls_content, client_profile
    )
    populated_veteran_ontology = await populate_veteran_ontology.execute()

    populated_template_workflow = PopulateFieTemplateWorkflow(
        config,
        ontology_content.base_template,
        client_profile,
        populated_veteran_ontology.value,
    )
    populated_template_completion = await populated_template_workflow.execute()
    if utils.is_completion_actionable(populated_template_completion.value):
        populated_template = populated_template_completion.value
    else:
        populated_template = populated_template_workflow.get_completion_by_name(
            "populated_template_first_pass"
        ).value

    ai_generated_content = utils.inject_identifiers(
        populated_template, full_name, ssn)
    if ontology_content.static_contents:
        ai_generated_content = "\n\n".join(
            (ai_generated_content, *ontology_content.static_contents)
        )

    return chat_types.GeneratedDraftParts(
        ai_generated_content=ai_generated_content,
        intermediary_assets=[
            extractors.extract_intended_type_from_completion(
                populated_veteran_ontology.value,
                types.ChatCompletionType.JSON,
                "populated_veteran_ontology",
            )
        ],
    )
