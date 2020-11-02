import React, { Component, useCallback } from "react";
import Dropzone from "react-dropzone";
import Card from "react-bootstrap/Card";
import "./Uploader.css";

const getClassname = (className, isDragActive) => {
  if (!isDragActive) return className;
  return `${className} ${className}-active`;
};

export default class DnDImgUploader extends Component {
  constructor(props) {
    super(props);
    this.onDropURLs = props.onDropURLs;
    this.state = {
      images: [],
      imageUrls: [],
    };
  }

  getFilesRender() {
    return this.state.imageUrls.map((url) => (
      <img className="prevImg" key={url} src={url} />
    ));
  }

  getFilesURLs(acceptedFiles) {
    return acceptedFiles.map((file) => {
      //   console.log("Creating url for " + file);
      return URL.createObjectURL(file);
    });
  }

  componentWillUnmount() {
    console.log("unmounting, revoking image urls");
    this.state.imageUrls.forEach((url) => URL.revokeObjectURL(url));
  }

  render() {
    const onDrop = (acceptedFiles) => {
      const urls = this.getFilesURLs(acceptedFiles);
      this.onDropURLs(urls);
      this.setState((state, props) => {
        return {
          images: [...state.images, ...acceptedFiles],
          imageUrls: [...state.imageUrls, ...urls],
        };
      });
    };

    return (
      <Dropzone onDrop={onDrop} accept={"image/*"}>
        {({ getRootProps, getInputProps, isDragActive }) => (
          <Card
            className={getClassname("dropzone", isDragActive)}
            {...getRootProps()}
            body
          >
            <div className="imageList">
              {this.getFilesRender()}
              <input className="dropzone-input" {...getInputProps()} />
              <div className="text-center">
                {isDragActive ? (
                  <p className="dropzone-content">Release files here</p>
                ) : (
                  <p className="dropzone-content">
                    Drag 'n' drop files here, or click to select files
                  </p>
                )}
              </div>
            </div>
          </Card>
        )}
      </Dropzone>
    );
  }
}
