from rest_framework import serializers

class AskQuestionSerializer(serializers.Serializer):
    prompt = serializers.CharField()
    config = serializers.JSONField()
    pdf_paths = serializers.ListField(child=serializers.CharField())