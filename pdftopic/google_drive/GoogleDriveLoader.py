import io
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import googleapiclient.discovery
from typing import List


class GoogleDriveLoader:
    VALID_EXTENSIONS = [".pdf"]

    def __init__(self, service: googleapiclient.discovery.Resource):

        self.service = service

    def search_for_files(self) -> List:
        """
        See https://developers.google.com/drive/api/guides/search-files#python
        """

        query = "mimeType != 'application/vnd.google-apps.folder' and ("
        for i, ext in enumerate(self.VALID_EXTENSIONS):
            if i == 0:
                query += "name contains '{}' ".format(ext)
            else:
                query += "or name contains '{}' ".format(ext)
        query = query.rstrip()
        query += ")"

        # create drive api client
        files = []
        page_token = None
        try:
            while True:
                # pylint: disable=maybe-no-member
                response = (
                    self.service.files()
                    .list(
                        q=query,
                        spaces="drive",
                        fields="nextPageToken, files(id, name)",
                        pageToken=page_token,
                    )
                    .execute()
                )
                for file in response.get("files"):
                    # Process change
                    print(f'Found file: {file.get("name")}, {file.get("id")}')

                    file_id = file.get("id")
                    file_name = file.get("name")

                    files.append(
                        {
                            "id": file_id,
                            "name": file_name,
                        }
                    )

                page_token = response.get("nextPageToken", None)
                if page_token is None:
                    break

        except HttpError as error:
            print(f"An error occurred: {error}")
            files = None

        return files

    def download_file(self, real_file_id: str) -> bytes:
        """
        Downloads a file
        """

        try:
            file_id = real_file_id
            request = self.service.files().get_media(fileId=file_id)
            file = io.BytesIO()
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}.")

        except HttpError as error:
            print(f"An error occurred: {error}")
            file = None

        return file.getvalue()
