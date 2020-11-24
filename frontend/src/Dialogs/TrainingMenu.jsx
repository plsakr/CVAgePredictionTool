"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __spreadArrays = (this && this.__spreadArrays) || function () {
    for (var s = 0, i = 0, il = arguments.length; i < il; i++) s += arguments[i].length;
    for (var r = Array(s), k = 0, i = 0; i < il; i++)
        for (var a = arguments[i], j = 0, jl = a.length; j < jl; j++, k++)
            r[k] = a[j];
    return r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var React = require("react");
var Modal_1 = require("react-bootstrap/Modal");
require("../Dialog.css");
var Button_1 = require("react-bootstrap/Button");
var ButtonGroup_1 = require("react-bootstrap/ButtonGroup");
var ToggleButton_1 = require("react-bootstrap/ToggleButton");
var Params_1 = require("../Training/Params");
var Dataset_1 = require("../Training/Dataset");
var Config_1 = require("../Config");
var TrainingMenu = /** @class */ (function (_super) {
    __extends(TrainingMenu, _super);
    function TrainingMenu(props) {
        var _this = _super.call(this, props) || this;
        _this.state = {
            isOpen: false,
            tabValue: "0",
            minK: 1,
            maxK: 100,
            oldPicsNbr: 10,
            youngPicsNbr: 10,
            testingRatio: 0.2,
            isChecked: false,
            youngUrls: [],
            oldUrls: [],
        };
        _this.onTrain = props.onTrain;
        return _this;
    }
    // allow others to open me
    TrainingMenu.prototype.open = function () {
        this.setState({ isOpen: true });
    };
    TrainingMenu.prototype.resolveURLs = function (urls) {
        return new Promise(function (resolve, reject) {
            console.log(urls);
            var count = urls.length;
            var result = [];
            urls.forEach(function (url) {
                var reader = new FileReader();
                var blob = fetch(url).then(function (r) {
                    return r.blob().then(function (blob) {
                        reader.readAsDataURL(blob);
                        reader.onload = function () {
                            result.push(reader.result);
                            if (result.length == count) {
                                resolve(result);
                            }
                        };
                    });
                });
            });
        });
    };
    TrainingMenu.prototype.performRequest = function (req) {
        var _this = this;
        if (req !== "undefined") {
            fetch(Config_1.backend + "/train", req).then(function (res) {
                if (!res.ok) {
                    console.log("There was a problem with the train request!");
                }
                else {
                    res.json().then(function (body) {
                        var jobId = body.jobId;
                        console.log("MY TRAINING ID IS " + jobId);
                        _this.setState({ isOpen: false });
                        _this.onTrain(jobId);
                    });
                }
            });
        }
    };
    // train button clicked. gather all data, tell the app what happened, and close the dialog
    TrainingMenu.prototype.handleOnTrain = function () {
        return __awaiter(this, void 0, void 0, function () {
            var minKVal, maxKVal, isChecked, oldPicsNbr, youngPicsNbr, testingRatio, req, req, minKVal_1, maxKVal_1, isChecked, testingRatio_1, youngUrls, oldUrls_1;
            var _this = this;
            return __generator(this, function (_a) {
                console.log("dialog train button pressed");
                if (this.state.tabValue === "0") {
                    minKVal = this.state.minK;
                    maxKVal = this.state.maxK;
                    isChecked = this.state.isChecked;
                    oldPicsNbr = this.state.oldPicsNbr;
                    youngPicsNbr = this.state.youngPicsNbr;
                    testingRatio = this.state.testingRatio;
                    if (isChecked) {
                        if (maxKVal > minKVal) {
                            req = {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                    isReset: false,
                                    isCustom: false,
                                    optimizeK: true,
                                    minK: minKVal,
                                    maxK: maxKVal,
                                    nbrYoung: youngPicsNbr,
                                    nbrOld: oldPicsNbr,
                                    testRatio: testingRatio,
                                }),
                            };
                            this.performRequest(req);
                        }
                        else {
                            console.log("maxK cannot be larger than minK");
                        }
                    }
                    else {
                        req = {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                isReset: false,
                                isCustom: false,
                                optimizeK: false,
                                minK: minKVal,
                                nbrYoung: youngPicsNbr,
                                nbrOld: oldPicsNbr,
                                testRatio: testingRatio,
                            }),
                        };
                        this.performRequest(req);
                    }
                }
                else {
                    minKVal_1 = this.state.minK;
                    maxKVal_1 = this.state.maxK;
                    isChecked = this.state.isChecked;
                    testingRatio_1 = this.state.testingRatio;
                    youngUrls = this.state.youngUrls;
                    oldUrls_1 = this.state.oldUrls;
                    if (isChecked) {
                        if (maxKVal_1 > minKVal_1) {
                            this.resolveURLs(youngUrls).then(function (youngData) {
                                _this.resolveURLs(oldUrls_1).then(function (oldData) {
                                    var body = {
                                        isReset: false,
                                        isCustom: true,
                                        optimizeK: true,
                                        minK: minKVal_1,
                                        maxK: maxKVal_1,
                                        youngPics: youngData,
                                        oldPics: oldData,
                                        testRatio: testingRatio_1,
                                    };
                                    var req = {
                                        method: "POST",
                                        headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify(body),
                                    };
                                    _this.performRequest(req);
                                });
                            });
                        }
                    }
                    else {
                        this.resolveURLs(youngUrls).then(function (youngData) {
                            _this.resolveURLs(oldUrls_1).then(function (oldData) {
                                var req = {
                                    method: "POST",
                                    headers: { "Content-Type": "application/json" },
                                    body: JSON.stringify({
                                        isReset: false,
                                        isCustom: true,
                                        optimizeK: false,
                                        minK: minKVal_1,
                                        youngPics: youngData,
                                        oldPics: oldData,
                                        testRatio: testingRatio_1,
                                    }),
                                };
                                _this.performRequest(req);
                            });
                        });
                    }
                }
                return [2 /*return*/];
            });
        });
    };
    // close button clicked. close dialog and do nothing
    TrainingMenu.prototype.handleOnClose = function () {
        this.setState({
            isOpen: false,
            isChecked: false,
            tabValue: "0",
            minK: 0,
            maxK: 100,
            oldPicsNbr: 1,
            youngPicsNbr: 1,
            testingRatio: 0.2,
            oldUrls: [],
            youngUrls: [],
        });
    };
    TrainingMenu.prototype.handleFormChange = function (e, isCheckbox) {
        if (isCheckbox === void 0) { isCheckbox = false; }
        var name = isCheckbox ? "isChecked" : e.target.name;
        var value = isCheckbox ? e.target.checked : Number(e.target.value);
        console.log(name);
        console.log(value);
        // this.setState({
        //   [name]: value,
        // });
    };
    TrainingMenu.prototype.handleYoungUpload = function (urls) {
        this.setState(function (state, props) {
            return {
                youngUrls: __spreadArrays(state.youngUrls, urls),
            };
        });
    };
    TrainingMenu.prototype.handleOldUpload = function (urls) {
        this.setState(function (state, props) {
            return {
                oldUrls: __spreadArrays(state.oldUrls, urls),
            };
        });
    };
    TrainingMenu.prototype.render = function () {
        var _this = this;
        // the tabs
        var radios = [
            { name: "Custom Parameters", value: "0" },
            { name: "Custom Dataset", value: "1" },
        ];
        // the tab content
        var currentState;
        if (this.state.tabValue == "0") {
            currentState = (<Params_1.default handleChange={this.handleFormChange.bind(this)} k={this.state.minK} maxK={this.state.maxK} oldPics={this.state.oldPicsNbr} youngPics={this.state.youngPicsNbr} testRatio={this.state.testingRatio} isChecked={this.state.isChecked}/>);
        }
        else {
            currentState = (<Dataset_1.default handleChange={this.handleFormChange.bind(this)} k={this.state.minK} maxK={this.state.maxK} testRatio={this.state.testingRatio} isChecked={this.state.isChecked} handleYoungUpload={this.handleYoungUpload.bind(this)} handleOldUpload={this.handleOldUpload.bind(this)}/>);
        }
        return (<Modal_1.default show={this.state.isOpen} onHide={function () { return _this.setState({ isOpen: false }); }} dialogClassName="modal-90w" aria-labelledby="example-custom-modal-styling-title" centered>
        <Modal_1.default.Header closeButton>
          <Modal_1.default.Title id="example-custom-modal-styling-title">
            Train New Model
          </Modal_1.default.Title>
        </Modal_1.default.Header>
        <Modal_1.default.Body>
          <ButtonGroup_1.default toggle>
            {radios.map(function (radio, idx) { return (<ToggleButton_1.default key={idx} type="radio" variant="primary" name="radio" value={radio.value} checked={_this.state.tabValue == radio.value} onChange={function (e) {
            return _this.setState({
                tabValue: radio.value,
                oldUrls: [],
                youngUrls: [],
            });
        }}>
                {radio.name}
              </ToggleButton_1.default>); })}
          </ButtonGroup_1.default>
          {currentState}
        </Modal_1.default.Body>
        <Modal_1.default.Footer>
          <Button_1.default variant="success" onClick={this.handleOnTrain.bind(this)}>
            Train
          </Button_1.default>
          <Button_1.default variant="outline-secondary" onClick={this.handleOnClose.bind(this)}>
            Close
          </Button_1.default>
        </Modal_1.default.Footer>
      </Modal_1.default>);
    };
    return TrainingMenu;
}(React.Component));
exports.default = TrainingMenu;
