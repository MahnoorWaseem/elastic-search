import io
import os
import uuid
import mimetypes
from pathlib import Path

import boto3
import cv2
from botocore.config import Config
from botocore.exceptions import ClientError
from config import (
    SEAWEED_ACCESS_KEY,
    SEAWEED_BUCKET,
    SEAWEED_S3_ENDPOINT,
    SEAWEED_SECRET_KEY,
)


class SeaweedClient:
    def __init__(self):
        self.bucket = SEAWEED_BUCKET

        self.s3 = boto3.client(
            "s3",
            endpoint_url=SEAWEED_S3_ENDPOINT,
            aws_access_key_id=SEAWEED_ACCESS_KEY,
            aws_secret_access_key=SEAWEED_SECRET_KEY,
            region_name="us-east-1",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )

    def ensure_bucket(self):
        """Create bucket if it doesn't exist."""
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")

            if status != 404:
                raise

            self.s3.create_bucket(Bucket=self.bucket)

    def upload_image(
        self,
        image_path: str,
        object_key: str | None = None,
    ) -> str:
        """Upload image from local filesystem."""

        path = Path(image_path)

        if object_key is None:
            object_key = f"images/{uuid.uuid4()}{path.suffix}"

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

        self.s3.upload_file(
            str(path),
            self.bucket,
            object_key,
            ExtraArgs={"ContentType": content_type},
        )

        return object_key

    def upload_bytes(
        self,
        image_bytes: bytes,
        object_key: str | None = None,
        content_type: str = "image/jpeg",
    ) -> str:
        """Upload raw image bytes."""

        if object_key is None:
            object_key = f"images/{uuid.uuid4()}.jpg"

        self.s3.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=image_bytes,
            ContentType=content_type,
        )

        return object_key

    def upload_frame(
        self,
        frame,
        object_key: str | None = None,
        quality: int = 90,
    ) -> str:
        """Encode and upload an OpenCV image/frame."""

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )

        if not success:
            raise ValueError("Failed to encode image")

        return self.upload_bytes(
            image_bytes=buffer.tobytes(),
            object_key=object_key,
        )

    def get_image(self, object_key: str) -> bytes:
        """Retrieve image as bytes."""

        response = self.s3.get_object(
            Bucket=self.bucket,
            Key=object_key,
        )

        return response["Body"].read()

    def download_image(
        self,
        object_key: str,
        destination: str,
    ):
        """Download image to local filesystem."""

        Path(destination).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.s3.download_file(
            self.bucket,
            object_key,
            destination,
        )

    def delete_image(self, object_key: str):
        """Delete an image."""

        self.s3.delete_object(
            Bucket=self.bucket,
            Key=object_key,
        )

    def generate_download_url(
        self,
        object_key: str,
        expires_in: int = 3600,
    ) -> str:
        """Generate temporary image access URL."""

        return self.s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self.bucket,
                "Key": object_key,
            },
            ExpiresIn=expires_in,
        )
