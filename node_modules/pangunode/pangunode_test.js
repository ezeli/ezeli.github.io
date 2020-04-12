'use strict';

var pangunode = require('./pangunode.js');

/*
  ======== A Handy Little Nodeunit Reference ========
  https://github.com/caolan/nodeunit

  Test methods:
    test.expect(numAssertions)
    test.done()
  Test assertions:
    test.ok(value, [message])
    test.equal(actual, expected, [message])
    test.notEqual(actual, expected, [message])
    test.deepEqual(actual, expected, [message])
    test.notDeepEqual(actual, expected, [message])
    test.strictEqual(actual, expected, [message])
    test.notStrictEqual(actual, expected, [message])
    test.throws(block, [error], [message])
    test.doesNotThrow(block, [error], [message])
    test.ifError(value)
*/

exports.pangunode = {
  setUp: function(done) {
    // setup here
    done();
  },
  'no args': function(test) {
    test.expect(1);
    // tests here
    test.strictEqual(pangunode(''), '', 'should be null.');
    test.done();
  },
  'args': function(test) {
    test.expect(3);
    // tests here
    test.strictEqual(pangunode('nothing happen'), 'nothing happen', 'should be \'nothing happen\'.');
    test.strictEqual(pangunode('應該nothing happen'), '應該 nothing happen', 'should be \'應該 nothing happen\'.');
    test.strictEqual(pangunode('u請問Jackie的鼻子有幾個S'), 'u 請問 Jackie 的鼻子有幾個 S', 'should be \'u 請問 Jackie 的鼻子有幾個 S\'.');
    test.done();
  }
};
