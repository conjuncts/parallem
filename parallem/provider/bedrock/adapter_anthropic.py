from parallem.provider.anthropic.adapter import AnthropicAdapter


class BedrockAnthropicAdapter(AnthropicAdapter):
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html
    def prepare_request(self, params, **kwargs):
        body = super().prepare_request(params, **kwargs)

        if "anthropic_version" not in body:
            body["anthropic_version"] = "bedrock-2023-05-31"
        if "model" in body:
            body.pop("model")  # modelId already provided
        return body
