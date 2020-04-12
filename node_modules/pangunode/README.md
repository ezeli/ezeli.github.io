# pangu.node [![Build Status](https://secure.travis-ci.org/huei90/pangu.node.png?branch=master)](http://travis-ci.org/huei90/pangu.node)

Spacing texts in Node.js!

[DEMO in browser](http://huei90.github.io/pangu.node/)

The algorithm is from [paranoid-auto-spacing](https://github.com/vinta/paranoid-auto-spacing)

## Getting Started
Install the module with: `npm install --save pangunode`

```javascript
var pangunode = require('pangunode');

pangunode('應該nothing happen');
// => 應該 nothing happen

pangunode('u請問Jackie的鼻子有幾個S');
// => u 請問 Jackie 的鼻子有幾個 S
```

## License
Copyright (c) 2014 Huei Tan. Licensed under the MIT license.
