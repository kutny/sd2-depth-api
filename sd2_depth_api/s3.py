def parse_s3_url(url):
    # example: https://s3.eu-west-1.amazonaws.com/some-bucket/foo/bar
    url_parts = url.split("/")
    key = "/".join(url_parts[4:])
    bucket_name = url_parts[3:][0]

    return bucket_name, key
